emulator_name="emulator-5554"
console_port=5554
grpc=8554
setup=True

agent_name="VDroid"
lora_name="Round110k"
llm_name="gpt-4o"

# summar=rule
summary=llm
save_name="Round110k_try"

text_name="./saved/${agent_name}_${save_name}/full_log.txt"

mkdir -p "$(dirname "$text_name")"

CUDA_VISIBLE_DEIVCES=0 python run.py --agent_name=$agent_name --lora_dir "RewardModel_$lora_name" --llm_name=$llm_name --summary=$summary --save_name=$save_name --device_name=$emulator_name --console_port=$console_port --grpc_port=$grpc --perform_emulator_setup=$setup >>$text_name
