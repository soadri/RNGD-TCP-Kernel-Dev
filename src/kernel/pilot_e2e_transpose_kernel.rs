use furiosa_opt_std::prelude::*;

// [M, N] f32 → [N, M] f32
// verify_transpose::<f32, m![B], m![C # 8], m![C], m![B # 8]>() 패턴
// M=B=2 (rows), N=C=8 (cols)
axes![M = 2, N = 8];

pub type Chip    = m![1];
pub type Cluster = m![1 # 2];
pub type Slice   = m![1 # 256];  // gemm처럼 Slice=1, M/N을 Time/Packet으로 처리

#[device(chip = 1)]
pub fn pilot_e2e_transpose_kernel(
    ctx: &mut Context,
    input: &HbmTensor<f32, Chip, m![M, N]>,
) -> HbmTensor<f32, Chip, m![N, M]> {
    let input_dm = input.to_dm::<Cluster, Slice, m![M, N]>(&mut ctx.tdma, 0);

    let result: DmTensor<f32, Chip, Cluster, Slice, m![N, M]> = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![M], m![N]>()
        .collect::<m![M], m![N]>()
        .transpose::<m![N], m![M # 8]>()
        .commit_trim::<m![M]>()
        .commit(1 << 13);

    result.to_hbm(&mut ctx.tdma, 1 << 28)
}
