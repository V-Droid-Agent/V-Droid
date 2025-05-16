# default configs for the eveluation on androidworld benchmark
emulator_name="emulator-5554" # emulator name 
console_port=5554 # the port for the console communications
grpc=8554 # the port for the accessibility and grpc service
setup=True # whether to perform the emulator and evelaution env setup

agent_name="VDroid" # the agent name used for the eveluation, e.g., default t3a, m3a in androidworld or VDroid
lora_name="Round110k" # the name of the folder where the lora weights of VDroid is saved
summary=llm # the mode for the working memory construction
llm_name="gpt-4o" # the llm used for the task completion and working memory construction
service_name="trapi" # the name of the service used for calling the llm above

save_name="Round110k_try" # the saved file name for exp data

mkdir -p "$(dirname "$text_name")"

CUDA_VISIBLE_DEIVCES=0 python run_suite.py \ 
    --agent_name=$agent_name \ 
    --lora_dir "RewardModel_$lora_name" \ 
    --llm_name=$llm_name \ 
    --service_name=$service_name \ 
    --summary=$summary \ 
    --save_name=$save_name \ 
    --device_name=$emulator_name \ 
    --console_port=$console_port \ 
    --grpc_port=$grpc \  
    --perform_emulator_setup=$setup