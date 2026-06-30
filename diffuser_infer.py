import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from diffusers import LTX2Pipeline

# 兼容 NPU + elementwise_affine=False 的 RMSNorm bug
# (diffusers RMSNorm.forward 在 NPU 分支未处理 weight=None 的情形,
#  会把 None 传给 npu_rms_norm 的 gamma 形参导致报错)
from diffusers.models.normalization import RMSNorm


def _patched_rms_norm_forward(self, hidden_states):
    from diffusers.utils.import_utils import is_torch_npu_available

    if is_torch_npu_available() and self.weight is not None:
        import torch_npu

        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)
        hidden_states = torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.eps)[0]
        if self.bias is not None:
            hidden_states = hidden_states + self.bias
        return hidden_states

    # weight is None 或非 NPU:纯 PyTorch RMSNorm
    input_dtype = hidden_states.dtype
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
    hidden_states = hidden_states.to(input_dtype)
    return hidden_states


RMSNorm.forward = _patched_rms_norm_forward
from diffusers.pipelines.ltx2.export_utils import encode_video
from diffusers.pipelines.ltx2.utils import DEFAULT_NEGATIVE_PROMPT

pipe = LTX2Pipeline.from_pretrained(
    "/home/wjh/ltx2.3-diffusers", torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()

prompt = "A flowing river in a forest at golden hour, gentle wind in the leaves."
frame_rate = 16.0

video, audio = pipe(
    prompt=prompt,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    width=832,
    height=480,
    num_frames=81,
    frame_rate=frame_rate,
    num_inference_steps=30,
    guidance_scale=3.0,
    output_type="np",
    return_dict=False,
)

encode_video(
    video[0],
    fps=frame_rate,
    audio=audio[0].float().cpu(),
    audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
    output_path="ltx2_t2v.mp4",
)
