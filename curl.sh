curl -X POST http://localhost:8099/v1/videos \
  -F "prompt='A flowing river in a forest at golden hour, gentle wind in the leaves.'" \
  -F "size=832x480" \
  -F "num_frames=81" \
  -F "fps=16" \
  -F "num_inference_steps=30" \
  -F "guidance_scale=3.0" \
  -F "seed=42" 

# -F "negative_prompt=low quality, blurry, static" \
# -F "input_reference=@/home/wjh/vllm-omni/input.jpg" \