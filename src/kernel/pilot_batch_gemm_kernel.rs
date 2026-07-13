use furiosa_opt_std::prelude::*;

// AUTO-GENERATED from rngd.batch_gemm — 최초 시도, 컴파일 에러 가능성 있음
// V=배치, M×K @ K×N -> M×N (문서의 bmatmul_m_in_time 전략: M in Time, K in Packet)
axes![V = 32, M = 32, K = 32, N = 8];

pub type Chip = m![1];
pub type Cluster = m![1];
// V=32개 배치만 실제 사용하지만, 하드웨어 제약(Slice는 64/128/192/256 중 하나)에
// 맞춰 256-slice 주소공간에 매핑 (dot_product_kernel.rs의 "1 # 256" 패턴과 동일 원리)
pub type Slice = m![V # 256];
pub type Lane = m![N];

#[device(chip = 1)]
pub fn pilot_batch_gemm_kernel(
    ctx: &mut Context,
    a: &HbmTensor<bf16, Chip, m![V, M, K]>,
    b: &HbmTensor<bf16, Chip, m![V, K, N]>,
) -> HbmTensor<bf16, Chip, m![V, M, N]> {
    let a: DmTensor<bf16, Chip, Cluster, Slice, m![M, K]> = a.to_dm(&mut ctx.tdma, 0);
    let b: DmTensor<bf16, Chip, Cluster, Slice, m![K, N]> = b.to_dm(&mut ctx.tdma, 1 << 12);

    // b(가중치)를 TRF에 미리 적재: N을 Lane에, K를 Element에
    let b_trf: TrfTensor<bf16, Chip, Cluster, Slice, Lane, m![K]> = ctx
        .sub
        .begin(b.view())
        .fetch::<m![N], m![K]>()
        .collect::<m![N, K / 16], m![K % 16]>()
        .to_trf(TrfAddress::Full);

    let result: DmTensor<bf16, Chip, Cluster, Slice, m![M, N]> = ctx
        .main
        .begin(a.view())
        .fetch::<m![M], m![K]>()
        .collect::<m![M, K / 16], m![K % 16]>()
        .contract_outer::<m![M], m![K], _, _>(&b_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![M]>()
        .contract_lane::<m![M], m![N]>(LaneMode::Interleaved)
        .cast::<bf16, m![N # 16]>()
        .commit_trim::<m![N]>()
        .commit(0);

    result.to_hbm(&mut ctx.tdma, 2 << 28)
}
