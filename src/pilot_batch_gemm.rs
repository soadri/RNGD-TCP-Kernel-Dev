use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_batch_gemm_kernel::{V, M, K, N, pilot_batch_gemm_kernel};
use rand::SeedableRng;
use rand::rngs::SmallRng;

#[tokio::main]
async fn main() {
    let mut ctx = Context::acquire();
    let mut rng = SmallRng::seed_from_u64(42);
    let a = HostTensor::<bf16, m![V, M, K]>::rand(&mut rng);
    let b = HostTensor::<bf16, m![V, K, N]>::rand(&mut rng);
    let a_hbm = a.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let b_hbm = b.to_hbm(&mut ctx.pdma, 1 << 28).await;
    let _out_hbm = launch(pilot_batch_gemm_kernel, (&mut ctx, &a_hbm, &b_hbm)).await;
    println!("Pilot Batch GEMM: kernel ran");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn matches_reference() {
        let mut ctx = Context::acquire();
        let mut rng = SmallRng::seed_from_u64(42);
        let a = HostTensor::<bf16, m![V, M, K]>::rand(&mut rng);
        let b = HostTensor::<bf16, m![V, K, N]>::rand(&mut rng);

        let a_hbm = a.to_hbm(&mut ctx.pdma, 0 << 28).await;
        let b_hbm = b.to_hbm(&mut ctx.pdma, 1 << 28).await;

        // Reference: 배치별 C[v,m,n] = sum_k A[v,m,k] * B[v,k,n], f32 누산 후 bf16 반올림
        // (실제 gemm.rs 테스트와 동일한 계산/허용오차 방식)
        let a_buf: Vec<bf16> = a.to_buf();
        let b_buf: Vec<bf16> = b.to_buf();
        let v_size = V::SIZE;
        let m_size = M::SIZE;
        let k_size = K::SIZE;
        let n_size = N::SIZE;

        let mut expected: Vec<bf16> = Vec::with_capacity(v_size * m_size * n_size);
        for v in 0..v_size {
            let a_batch = &a_buf[v * m_size * k_size..(v + 1) * m_size * k_size];
            let b_batch = &b_buf[v * k_size * n_size..(v + 1) * k_size * n_size];
            for m in 0..m_size {
                for n in 0..n_size {
                    let acc: f32 = (0..k_size)
                        .map(|k| f32::from(a_batch[m * k_size + k]) * f32::from(b_batch[k * n_size + n]))
                        .sum();
                    expected.push(bf16::from_f32(acc));
                }
            }
        }

        let out_hbm = launch(pilot_batch_gemm_kernel, (&mut ctx, &a_hbm, &b_hbm)).await;
        let actual: Vec<bf16> = out_hbm.to_host::<m![V, M, N]>(&mut ctx.pdma).await.to_buf();

        println!("=== 값 비교 (앞 8개) ===");
        for i in 0..8 {
            println!("  [{i}]: {:?} | {:?}", expected[i], actual[i]);
        }

        for (idx, (&e, &av)) in expected.iter().zip(&actual).enumerate() {
            let diff = (f32::from(av) - f32::from(e)).abs();
            let tol = (0.05 * f32::from(e).abs()).max(1.0);
            assert!(diff <= tol, "batch_gemm mismatch at idx={idx}: expected {e:?}, actual {av:?}");
        }
    }
}
