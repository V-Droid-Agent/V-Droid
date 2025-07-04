<div align=center>
   <img src="https://github.com/user-attachments/assets/fa805972-efdf-449d-a716-68364bbaaf93" width=600 height=400>
</div>

# :alien:V-Droid: Advancing Mobile GUI Agent Through Generative Verifiers

This repo provides the public preview for **V-Droid**(https://arxiv.org/abs/2503.15937), a verifier-driven mobile GUI agents. Unlike previous mobile agents that utilize Large Language Models (LLMs) as generators to directly generate actions at each step, V-Droid **employs LLMs** as verifiers to evaluate candidate actions before making final decisions. To realize this novel paradigm, we introduce a comprehensive framework for constructing verifier-driven mobile agents: the discretized *action space construction* coupled with the prefilling-only workflow to accelerate the verification process, *the pair-wise progress preference training* to significantly enhance the verifier's decision-making capabilities, and *the scalable human-agent joint annotation scheme* to efficiently collect the necessary data at scale. V-Droid sets a new state-of-the-art task success rate across several public mobile task automation benchmarks: 59.5% on AndroidWorld, 38.3% on AndroidLab, and 49% on MobileAgentBench, surpassing existing agents by 9.5%, 2.1%, and 9%, respectively. Furthermore, V-Droid achieves an impressively low latency of 0.7 seconds per step, making it the first mobile agent capable of delivering near-real-time, effective decision-making capabilities.

- :white_check_mark: Paper link: https://arxiv.org/abs/2503.15937
- :white_check_mark: Model weitghs: https://huggingface.co/V-Droid/V-Droid-8B-0323
- 
## Demos
V-Droid in the following demos are hosted on 2x4090 GPUs, the videos are presented without acceleration.

Delete the recipes from Broccoli app: Chicken Alfredo Pasta, Tomato Basil Bruschetta, Grilled Cheese with Tomato and Basil. | Swich on WiFi for me.. | Send a text message to +16597910719 with message: Beauty is in the eye of the beholder.
:--:|:--:|:--:
<img src="https://github.com/user-attachments/assets/9a69a239-7e3b-491b-a015-f507b6ca7463" width=200> | <img src="https://github.com/user-attachments/assets/6da1a714-d75c-428a-a450-e50234bf48c6" width=200> | <img src="https://github.com/user-attachments/assets/66be8f36-a3e3-4d01-b60d-6029777337e7" width=200>


## V-Droid Workflow

In V-Droid, we propose the verifier-driven approach and the correpsonding workflow for GUI agents as follows:

<div align=center>
   <img src="https://github.com/user-attachments/assets/47ea5579-ff2c-4f73-9f89-f0cabe9bbea6" width=600 height=400>
</div>

 1) Extracting actions from UI and supplementing default actions; 
 2) Constructing verification prompts with the template for each candidate action; 
 3) Scoring with the verifier in batch with prefix caching; 
 4) Completing and executing the selected action; 
 5) Updating the working memory.
For more details, please refer our code


## Quick Start
1. Setup AndroidWorld Environment
   1. Download Android Studio [here](https://developer.android.com/studio?gad_source=1&gclid=Cj0KCQjw3ZayBhDRARIsAPWzx8oLcadBD0vAq8xmUutaunLGSzhgEtLz4xVZ_SpV4G0xJazS7LxQkDsaAuveEALw_wcB&gclsrc=aw.ds)
   2. Create an Android Virtual Device (AVD) by following these instructions. For hardware select **Pixel 6**, for System Image select **Tiramisu, API Level 33**, and choose AVD name as **AndroidWorldAvd**. [Watch the setup video.](https://github.com/google-research/android_world/assets/162379927/efc33980-8b36-44be-bb2b-a92d4c334a50)

2. Launch the Android Emulator from the command line
    Launch the emulator from the command line, not using the Android Studio UI, with the `-grpc 8554` flag which is needed communication with accessibility forwarding app.

    ```bash
    # Typically it's located in ~/Android/Sdk/emulator/emulator or
    # ~/Library/Android/sdk/emulator/emulator
    EMULATOR_NAME=AndroidWorldAvd # From previous step
    ~/Library/Android/sdk/emulator/emulator -avd $EMULATOR_NAME -no-snapshot -grpc 8554
    ```

3. [Optional] It's recommended to use `conda`, which you can download [here](https://docs.anaconda.com/free/miniconda/miniconda-install/).

    ```
    conda create -n android_world python=3.11.8
    conda activate android_world
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    conda install -y numpy pandas
    ```

4. Install Dependency. *Note: Python 3.11 or above is required.*

    ```python
    pip install -r requirements.txt
    ```

5. Modify vLLM.
     
    Please navigate to vllm/model_executor/layers/sampler.py, add the following to line 317.

    ```python
    for val, lst in zip(logits, sample_logprobs):
            for d in lst:
                for k in d.keys():
                    d[k].logprob = val
    ```
    (See https://github.com/vllm-project/vllm/issues/11397 for more explanations)

6. Add model provider APIs as environment variables.

    Three API providers are supported: OpenAI and its compatible APIs, and Azure OpenAI services. You may configure any of these based on your preferences.
    ```bash
    # Add to .bashrc.

    # use Gemini GCP service, which requires API key
    export GCP_API_KEY=
    
    # use openai compatible APIs, including OPENAI, Qwen and DeepSeek
    export OPENAI_ENDPOINT=
    export OPENAI_MODEL_NAME=
    export OPENAI_API_VERSION=
    export OPENAI_API_KEY=


    # use azure openai services
    export AZURE_OPENAI_API_KEY=
    export AZURE_OPENAI_MODEL_NAME=
    export AZURE_OPENAI_API_VERSION=
    export AZURE_OPENAI_ENDPOINT=
    ```

7. Download Lora weights for V-Droid model
   
   The V-Droid model weight is available at https://huggingface.co/V-Droid/V-Droid-8B-0323

8. Lauanch the emulator and run the eveluation tasks
   ```bash
   emulator -avd AndroidWorldAvd -no-window -no-snapshot -grpc 8554
   bash main.sh
   ```


9. Training 
   You may use the following code to train the lora module in V-Droid. We provide several training pairs to use.
   ```bash
   train.sh 
   ```

## Citation
If you use this repo, please cite our paper:

```bibtex
@misc{dai2025advancingmobileguiagents,
      title={Advancing Mobile GUI Agents: A Verifier-Driven Approach to Practical Deployment}, 
      author={Gaole Dai and Shiqi Jiang and Ting Cao and Yuanchun Li and Yuqing Yang and Rui Tan and Mo Li and Lili Qiu},
      year={2025},
      eprint={2503.15937},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2503.15937}, 
}
```
