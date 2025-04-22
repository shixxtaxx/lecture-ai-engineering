{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "gSpnWBP5ELSI"
      },
      "source": [
        "# 実践演習 Day 1：streamlitとFastAPIのデモ\n",
        "このノートブックでは以下の内容を学習します。\n",
        "\n",
        "- 必要なライブラリのインストールと環境設定\n",
        "- Hugging Faceからモデルを用いたStreamlitのデモアプリ\n",
        "- FastAPIとngrokを使用したAPIの公開方法\n",
        "\n",
        "演習を始める前に、HuggingFaceとngrokのアカウントを作成し、\n",
        "それぞれのAPIトークンを取得する必要があります。\n",
        "\n",
        "\n",
        "演習の時間では、以下の3つのディレクトリを順に説明します。\n",
        "\n",
        "1. 01_streamlit_UI\n",
        "2. 02_streamlit_app\n",
        "3. 03_FastAPI\n",
        "\n",
        "2つ目や3つ目からでも始められる様にノートブックを作成しています。\n",
        "\n",
        "復習の際にもこのノートブックを役立てていただければと思います。\n",
        "\n",
        "### 注意事項\n",
        "「02_streamlit_app」と「03_FastAPI」では、GPUを使用します。\n",
        "\n",
        "これらを実行する際は、Google Colab画面上のメニューから「編集」→ 「ノートブックの設定」\n",
        "\n",
        "「ハードウェアアクセラレーター」の項目の中から、「T4 GPU」を選択してください。\n",
        "\n",
        "このノートブックのデフォルトは「CPU」になっています。\n",
        "\n",
        "---"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "OhtHkJOgELSL"
      },
      "source": [
        "# 環境変数の設定（1~3共有）\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Y-FjBp4MMQHM"
      },
      "source": [
        "GitHubから演習用のコードをCloneします。"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "AIXMavdDEP8U",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "1d767c38-476c-4c52-e74c-81caeb9c4d08"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Cloning into 'lecture-ai-engineering'...\n",
            "remote: Enumerating objects: 41, done.\u001b[K\n",
            "remote: Counting objects: 100% (32/32), done.\u001b[K\n",
            "remote: Compressing objects: 100% (27/27), done.\u001b[K\n",
            "remote: Total 41 (delta 7), reused 5 (delta 5), pack-reused 9 (from 1)\u001b[K\n",
            "Receiving objects: 100% (41/41), 34.04 KiB | 590.00 KiB/s, done.\n",
            "Resolving deltas: 100% (7/7), done.\n"
          ]
        }
      ],
      "source": [
        "!git clone https://github.com/shixxtaxx/lecture-ai-engineering.git"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "XC8n7yZ_vs1K"
      },
      "source": [
        "必要なAPIトークンを.envに設定します。\n",
        "\n",
        "「lecture-ai-engineering/day1」の配下に、「.env_template」ファイルが存在しています。\n",
        "\n",
        "隠しファイルのため表示されていない場合は、画面左側のある、目のアイコンの「隠しファイルの表示」ボタンを押してください。\n",
        "\n",
        "「.env_template」のファイル名を「.env」に変更します。「.env」ファイルを開くと、以下のような中身になっています。\n",
        "\n",
        "\n",
        "```\n",
        "HUGGINGFACE_TOKEN=\"hf-********\"\n",
        "NGROK_TOKEN=\"********\"\n",
        "```\n",
        "ダブルクオーテーションで囲まれた文字列をHuggingfaceのアクセストークンと、ngrokの認証トークンで書き変えてください。\n",
        "\n",
        "それぞれのアカウントが作成済みであれば、以下のURLからそれぞれのトークンを取得できます。\n",
        "\n",
        "- Huggingfaceのアクセストークン\n",
        "https://huggingface.co/docs/hub/security-tokens\n",
        "\n",
        "- ngrokの認証トークン\n",
        "https://dashboard.ngrok.com/get-started/your-authtoken\n",
        "\n",
        "書き換えたら、「.env」ファイルをローカルのPCにダウンロードしてください。\n",
        "\n",
        "「01_streamlit_UI」から「02_streamlit_app」へ進む際に、CPUからGPUの利用に切り替えるため、セッションが一度切れてしまいます。\n",
        "\n",
        "その際に、トークンを設定した「.env」ファイルは再作成することになるので、その手間を減らすために「.env」ファイルをダウンロードしておくと良いです。"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Py1BFS5RqcSS"
      },
      "source": [
        "「.env」ファイルを読み込み、環境変数として設定します。次のセルを実行し、最終的に「True」が表示されていればうまく読み込めています。"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "id": "bvEowFfg5lrq",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "41d32395-e206-4bf4-fd64-3ad5fe9f5185"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting python-dotenv\n",
            "  Downloading python_dotenv-1.1.0-py3-none-any.whl.metadata (24 kB)\n",
            "Downloading python_dotenv-1.1.0-py3-none-any.whl (20 kB)\n",
            "Installing collected packages: python-dotenv\n",
            "Successfully installed python-dotenv-1.1.0\n",
            "/content/lecture-ai-engineering/day1\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "True"
            ]
          },
          "metadata": {},
          "execution_count": 2
        }
      ],
      "source": [
        "!pip install python-dotenv\n",
        "from dotenv import load_dotenv, find_dotenv\n",
        "\n",
        "%cd /content/lecture-ai-engineering/day1\n",
        "load_dotenv(find_dotenv())"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "os0Yk6gaELSM"
      },
      "source": [
        "# 01_streamlit_UI\n",
        "\n",
        "ディレクトリ「01_streamlit_UI」に移動します。"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "id": "S28XgOm0ELSM",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "2d895f4c-6df0-4ce5-cd4a-1c7aee900b61"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "/content/lecture-ai-engineering/day1/01_streamlit_UI\n"
          ]
        }
      ],
      "source": [
        "%cd /content/lecture-ai-engineering/day1/01_streamlit_UI"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "eVp-aEIkELSM"
      },
      "source": [
        "必要なライブラリをインストールします。"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 5,
      "metadata": {
        "id": "nBe41LFiELSN"
      },
      "outputs": [],
      "source": [
        "%%capture\n",
        "!pip install -r requirements.txt"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Yyw6VHaTELSN"
      },
      "source": [
        "ngrokのトークンを使用して、認証を行います。"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 6,
      "metadata": {
        "id": "aYw1q0iXELSN",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "d022f989-8685-4573-807c-5cde640b08a0"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Authtoken saved to configuration file: /root/.config/ngrok/ngrok.yml\n"
          ]
        }
      ],
      "source": [
        "!ngrok authtoken $$NGROK_TOKEN"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "RssTcD_IELSN"
      },
      "source": [
        "アプリを起動します。"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 10,
      "metadata": {
        "id": "f-E7ucR6ELSN",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "2b944799-4e25-49fc-ea2c-668c233e397c"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "公開URL: https://6304-34-125-242-248.ngrok-free.app\n",
            "\n",
            "Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.\n",
            "\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m\u001b[1m  You can now view your Streamlit app in your browser.\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  Local URL: \u001b[0m\u001b[1mhttp://localhost:8501\u001b[0m\n",
            "\u001b[34m  Network URL: \u001b[0m\u001b[1mhttp://172.28.0.12:8501\u001b[0m\n",
            "\u001b[34m  External URL: \u001b[0m\u001b[1mhttp://34.125.242.248:8501\u001b[0m\n",
            "\u001b[0m\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "WARNING:pyngrok.process.ngrok:t=2025-04-22T17:08:04+0000 lvl=warn msg=\"failed to check for update\" obj=updater err=\"Post \\\"https://update.equinox.io/check\\\": context deadline exceeded\"\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[34m  Stopping...\u001b[0m\n",
            "\u001b[34m  Stopping...\u001b[0m\n",
            "Exception ignored in atexit callback: <function shutdown at 0x7fc0d8879440>\n",
            "Traceback (most recent call last):\n",
            "  File \"/usr/lib/python3.11/logging/__init__.py\", line 2194, in shutdown\n",
            "    h.release()\n",
            "  File \"/usr/lib/python3.11/logging/__init__.py\", line 929, in release\n",
            "    def release(self):\n",
            "\n",
            "  File \"/usr/local/lib/python3.11/dist-packages/streamlit/web/bootstrap.py\", line 44, in signal_handler\n",
            "    server.stop()\n",
            "  File \"/usr/local/lib/python3.11/dist-packages/streamlit/web/server/server.py\", line 470, in stop\n",
            "    self._runtime.stop()\n",
            "  File \"/usr/local/lib/python3.11/dist-packages/streamlit/runtime/runtime.py\", line 337, in stop\n",
            "    async_objs.eventloop.call_soon_threadsafe(stop_on_eventloop)\n",
            "  File \"/usr/lib/python3.11/asyncio/base_events.py\", line 807, in call_soon_threadsafe\n",
            "    self._check_closed()\n",
            "  File \"/usr/lib/python3.11/asyncio/base_events.py\", line 520, in _check_closed\n",
            "    raise RuntimeError('Event loop is closed')\n",
            "RuntimeError: Event loop is closed\n"
          ]
        }
      ],
      "source": [
        "from pyngrok import ngrok\n",
        "\n",
        "public_url = ngrok.connect(8501).public_url\n",
        "print(f\"公開URL: {public_url}\")\n",
        "!streamlit run app.py"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "kbYyXVFjELSN"
      },
      "source": [
        "公開URLの後に記載されているURLにブラウザでアクセスすると、streamlitのUIが表示されます。\n",
        "\n",
        "app.pyのコメントアウトされている箇所を編集することで、UIがどの様に変化するか確認してみましょう。\n",
        "\n",
        "streamlitの公式ページには、ギャラリーページがあります。\n",
        "\n",
        "streamlitを使うとpythonという一つの言語であっても、様々なUIを実現できることがわかると思います。\n",
        "\n",
        "https://streamlit.io/gallery"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "MmtP5GLOELSN"
      },
      "source": [
        "後片付けとして、使う必要のないngrokのトンネルを削除します。"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 11,
      "metadata": {
        "id": "8Ek9QgahELSO"
      },
      "outputs": [],
      "source": [
        "from pyngrok import ngrok\n",
        "ngrok.kill()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "o-T8tFpyELSO"
      },
      "source": [
        "# 02_streamlit_app"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "QqogFQKnELSO"
      },
      "source": [
        "\n",
        "ディレクトリ「02_streamlit_app」に移動します。"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 12,
      "metadata": {
        "id": "UeEjlJ7uELSO",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "5383f183-7dd6-4b34-c548-81332726c7e2"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "/content/lecture-ai-engineering/day1/02_streamlit_app\n"
          ]
        }
      ],
      "source": [
        "%cd /content/lecture-ai-engineering/day1/02_streamlit_app"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-XUH2AstELSO"
      },
      "source": [
        "必要なライブラリをインストールします。"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 14,
      "metadata": {
        "id": "mDqvI4V3ELSO"
      },
      "outputs": [],
      "source": [
        "%%capture\n",
        "!pip install -r requirements.txt"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ZO31umGZELSO"
      },
      "source": [
        "ngrokとhuggigfaceのトークンを使用して、認証を行います。"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 15,
      "metadata": {
        "id": "jPxTiEWQELSO",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "abf11e44-64eb-4f1d-f20e-bb157e783457"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Authtoken saved to configuration file: /root/.config/ngrok/ngrok.yml\n",
            "The token has not been saved to the git credentials helper. Pass `add_to_git_credential=True` in this function directly or `--add-to-git-credential` if using via `huggingface-cli` if you want to set the git credential as well.\n",
            "Token is valid (permission: write).\n",
            "The token `ai_engeneering` has been saved to /root/.cache/huggingface/stored_tokens\n",
            "Your token has been saved to /root/.cache/huggingface/token\n",
            "Login successful.\n",
            "The current active token is: `ai_engeneering`\n"
          ]
        }
      ],
      "source": [
        "!ngrok authtoken $$NGROK_TOKEN\n",
        "!huggingface-cli login --token $$HUGGINGFACE_TOKEN"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "dz4WrELLELSP"
      },
      "source": [
        "stramlitでHuggingfaceのトークン情報を扱うために、streamlit用の設定ファイル（.streamlit）を作成し、トークンの情報を格納します。"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 16,
      "metadata": {
        "id": "W184-a7qFP0W"
      },
      "outputs": [],
      "source": [
        "# .streamlit/secrets.toml ファイルを作成\n",
        "import os\n",
        "import toml\n",
        "\n",
        "# 設定ファイルのディレクトリ確保\n",
        "os.makedirs('.streamlit', exist_ok=True)\n",
        "\n",
        "# 環境変数から取得したトークンを設定ファイルに書き込む\n",
        "secrets = {\n",
        "    \"huggingface\": {\n",
        "        \"token\": os.environ.get(\"HUGGINGFACE_TOKEN\", \"\")\n",
        "    }\n",
        "}\n",
        "\n",
        "# 設定ファイルを書き込む\n",
        "with open('.streamlit/secrets.toml', 'w') as f:\n",
        "    toml.dump(secrets, f)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "fK0vI_xKELSP"
      },
      "source": [
        "アプリを起動します。\n",
        "\n",
        "02_streamlit_appでは、Huggingfaceからモデルをダウンロードするため、初回起動には2分程度時間がかかります。\n",
        "\n",
        "この待ち時間を利用して、app.pyのコードを確認してみましょう。"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 21,
      "metadata": {
        "id": "TBQyTTWTELSP",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "9d6230c8-82da-4a7f-9a13-db10234f5be0"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "公開URL: https://74a9-34-125-242-248.ngrok-free.app\n",
            "\n",
            "Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.\n",
            "\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m\u001b[1m  You can now view your Streamlit app in your browser.\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  Local URL: \u001b[0m\u001b[1mhttp://localhost:8501\u001b[0m\n",
            "\u001b[34m  Network URL: \u001b[0m\u001b[1mhttp://172.28.0.12:8501\u001b[0m\n",
            "\u001b[34m  External URL: \u001b[0m\u001b[1mhttp://34.125.242.248:8501\u001b[0m\n",
            "\u001b[0m\n",
            "NLTK loaded successfully.\n",
            "2025-04-22 17:25:31.268877: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered\n",
            "WARNING: All log messages before absl::InitializeLog() is called are written to STDERR\n",
            "E0000 00:00:1745342731.292697    7966 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered\n",
            "E0000 00:00:1745342731.299855    7966 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered\n",
            "2025-04-22 17:25:31.331335: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.\n",
            "To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.\n",
            "NLTK Punkt data checked/downloaded.\n",
            "Database 'chat_feedback.db' initialized successfully.\n",
            "Loading checkpoint shards: 100% 2/2 [00:00<00:00,  9.41it/s]\n",
            "Device set to use cuda\n",
            "2025-04-22 17:25:43.534 Examining the path of torch.classes raised:\n",
            "Traceback (most recent call last):\n",
            "  File \"/usr/local/lib/python3.11/dist-packages/streamlit/web/bootstrap.py\", line 347, in run\n",
            "    if asyncio.get_running_loop().is_running():\n",
            "       ^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
            "RuntimeError: no running event loop\n",
            "\n",
            "During handling of the above exception, another exception occurred:\n",
            "\n",
            "Traceback (most recent call last):\n",
            "  File \"/usr/local/lib/python3.11/dist-packages/streamlit/watcher/local_sources_watcher.py\", line 217, in get_module_paths\n",
            "    potential_paths = extract_paths(module)\n",
            "                      ^^^^^^^^^^^^^^^^^^^^^\n",
            "  File \"/usr/local/lib/python3.11/dist-packages/streamlit/watcher/local_sources_watcher.py\", line 210, in <lambda>\n",
            "    lambda m: list(m.__path__._path),\n",
            "                   ^^^^^^^^^^^^^^^^\n",
            "  File \"/usr/local/lib/python3.11/dist-packages/torch/_classes.py\", line 13, in __getattr__\n",
            "    proxy = torch._C._get_custom_class_python_wrapper(self.name, attr)\n",
            "            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
            "RuntimeError: Tried to instantiate class '__path__._path', but it does not exist! Ensure that it is registered via torch::class_\n",
            "\u001b[34m  Stopping...\u001b[0m\n",
            "\u001b[34m  Stopping...\u001b[0m\n",
            "^C\n"
          ]
        }
      ],
      "source": [
        "from pyngrok import ngrok\n",
        "\n",
        "public_url = ngrok.connect(8501).public_url\n",
        "print(f\"公開URL: {public_url}\")\n",
        "!streamlit run app.py"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "VBrOuWuEawqJ"
      },
      "source": [
        "アプリケーションの機能としては、チャット機能や履歴閲覧があります。\n",
        "\n",
        "これらの機能を実現するためには、StreamlitによるUI部分だけではなく、SQLiteを使用したチャット履歴の保存やLLMのモデルを呼び出した推論などの処理を組み合わせることで実現しています。\n",
        "\n",
        "- **`app.py`**: アプリケーションのエントリーポイント。チャット機能、履歴閲覧、サンプルデータ管理のUIを提供します。\n",
        "- **`ui.py`**: チャットページや履歴閲覧ページなど、アプリケーションのUIロジックを管理します。\n",
        "- **`llm.py`**: LLMモデルのロードとテキスト生成を行うモジュール。\n",
        "- **`database.py`**: SQLiteデータベースを使用してチャット履歴やフィードバックを保存・管理します。\n",
        "- **`metrics.py`**: BLEUスコアやコサイン類似度など、回答の評価指標を計算するモジュール。\n",
        "- **`data.py`**: サンプルデータの作成やデータベースの初期化を行うモジュール。\n",
        "- **`config.py`**: アプリケーションの設定（モデル名やデータベースファイル名）を管理します。\n",
        "- **`requirements.txt`**: このアプリケーションを実行するために必要なPythonパッケージ。"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Xvm8sWFPELSP"
      },
      "source": [
        "後片付けとして、使う必要のないngrokのトンネルを削除します。"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 22,
      "metadata": {
        "id": "WFJC2TmZELSP"
      },
      "outputs": [],
      "source": [
        "from pyngrok import ngrok\n",
        "ngrok.kill()"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "rUXhIzV7ELSP"
      },
      "source": [
        "# 03_FastAPI\n",
        "\n",
        "ディレクトリ「03_FastAPI」に移動します。"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 23,
      "metadata": {
        "id": "4ejjDLxr3kfC",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "04afca62-bc2d-4c96-9f5d-15741ecacfb4"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "/content/lecture-ai-engineering/day1/03_FastAPI\n"
          ]
        }
      ],
      "source": [
        "%cd /content/lecture-ai-engineering/day1/03_FastAPI"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "f45TDsNzELSQ"
      },
      "source": [
        "必要なライブラリをインストールします。"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 24,
      "metadata": {
        "id": "9uv6glCz5a7Z"
      },
      "outputs": [],
      "source": [
        "%%capture\n",
        "!pip install -r requirements.txt"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "JfrmE2VmELSQ"
      },
      "source": [
        "ngrokとhuggigfaceのトークンを使用して、認証を行います。"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 25,
      "metadata": {
        "id": "ELzWhMFORRIO",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "bbd8dd64-f45e-4c3f-eccf-243b554a5167"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Authtoken saved to configuration file: /root/.config/ngrok/ngrok.yml\n",
            "The token has not been saved to the git credentials helper. Pass `add_to_git_credential=True` in this function directly or `--add-to-git-credential` if using via `huggingface-cli` if you want to set the git credential as well.\n",
            "Token is valid (permission: write).\n",
            "The token `ai_engeneering` has been saved to /root/.cache/huggingface/stored_tokens\n",
            "Your token has been saved to /root/.cache/huggingface/token\n",
            "Login successful.\n",
            "The current active token is: `ai_engeneering`\n"
          ]
        }
      ],
      "source": [
        "!ngrok authtoken $$NGROK_TOKEN\n",
        "!huggingface-cli login --token $$HUGGINGFACE_TOKEN"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "t-wztc2CELSQ"
      },
      "source": [
        "アプリを起動します。\n",
        "\n",
        "「02_streamlit_app」から続けて「03_FastAPI」を実行している場合は、モデルのダウンロードが済んでいるため、すぐにサービスが立ち上がります。\n",
        "\n",
        "「03_FastAPI」のみを実行している場合は、初回の起動時にモデルのダウンロードが始まるので、モデルのダウンロードが終わるまで数分間待ちましょう。"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 28,
      "metadata": {
        "id": "meQ4SwISn3IQ",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "860f763c-692f-4c34-b02f-31e64b15f3b1"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "2025-04-22 17:35:04.798138: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered\n",
            "WARNING: All log messages before absl::InitializeLog() is called are written to STDERR\n",
            "E0000 00:00:1745343304.818348   10502 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered\n",
            "E0000 00:00:1745343304.824439   10502 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered\n",
            "2025-04-22 17:35:04.844679: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.\n",
            "To enable the following instructions: AVX2 AVX512F FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.\n",
            "モデル名を設定: google/gemma-3-1b-it\n",
            "/content/lecture-ai-engineering/day1/03_FastAPI/app.py:134: DeprecationWarning: \n",
            "        on_event is deprecated, use lifespan event handlers instead.\n",
            "\n",
            "        Read more about it in the\n",
            "        [FastAPI docs for Lifespan Events](https://fastapi.tiangolo.com/advanced/events/).\n",
            "        \n",
            "  @app.on_event(\"startup\")\n",
            "FastAPIエンドポイントを定義しました。\n",
            "アクティブなngrokトンネルはありません。\n",
            "ポート8501に新しいngrokトンネルを開いています...\n",
            "---------------------------------------------------------------------\n",
            "✅ 公開URL:   https://477c-34-125-242-248.ngrok-free.app\n",
            "📖 APIドキュメント (Swagger UI): https://477c-34-125-242-248.ngrok-free.app/docs\n",
            "---------------------------------------------------------------------\n",
            "(APIクライアントやブラウザからアクセスするためにこのURLをコピーしてください)\n",
            "\u001b[32mINFO\u001b[0m:     Started server process [\u001b[36m10502\u001b[0m]\n",
            "\u001b[32mINFO\u001b[0m:     Waiting for application startup.\n",
            "load_model_task: モデルの読み込みを開始...\n",
            "使用デバイス: cuda\n",
            "config.json: 100% 899/899 [00:00<00:00, 5.70MB/s]\n",
            "model.safetensors:  62% 1.24G/2.00G [00:09<00:03, 250MB/s]t=2025-04-22T17:35:19+0000 lvl=warn msg=\"failed to open private leg\" id=e1eee0b69d60 privaddr=localhost:8501 err=\"dial tcp 127.0.0.1:8501: connect: connection refused\"\n",
            "model.safetensors: 100% 2.00G/2.00G [00:12<00:00, 161MB/s]\n",
            "generation_config.json: 100% 215/215 [00:00<00:00, 1.46MB/s]\n",
            "tokenizer_config.json: 100% 1.16M/1.16M [00:00<00:00, 5.60MB/s]\n",
            "tokenizer.model: 100% 4.69M/4.69M [00:00<00:00, 315MB/s]\n",
            "tokenizer.json: 100% 33.4M/33.4M [00:00<00:00, 300MB/s]\n",
            "added_tokens.json: 100% 35.0/35.0 [00:00<00:00, 281kB/s]\n",
            "special_tokens_map.json: 100% 662/662 [00:00<00:00, 3.39MB/s]\n",
            "t=2025-04-22T17:35:28+0000 lvl=warn msg=\"failed to open private leg\" id=42887fb6e8a0 privaddr=localhost:8501 err=\"dial tcp 127.0.0.1:8501: connect: connection refused\"\n",
            "Device set to use cuda\n",
            "モデル 'google/gemma-3-1b-it' の読み込みに成功しました\n",
            "load_model_task: モデルの読み込みが完了しました。\n",
            "起動時にモデルの初期化が完了しました。\n",
            "\u001b[32mINFO\u001b[0m:     Application startup complete.\n",
            "\u001b[32mINFO\u001b[0m:     Uvicorn running on \u001b[1mhttp://0.0.0.0:8501\u001b[0m (Press CTRL+C to quit)\n",
            "\u001b[32mINFO\u001b[0m:     126.110.235.27:0 - \"\u001b[1mGET /docs HTTP/1.1\u001b[0m\" \u001b[32m200 OK\u001b[0m\n",
            "\u001b[32mINFO\u001b[0m:     126.110.235.27:0 - \"\u001b[1mGET /openapi.json HTTP/1.1\u001b[0m\" \u001b[32m200 OK\u001b[0m\n",
            "\u001b[32mINFO\u001b[0m:     Shutting down\n",
            "\u001b[32mINFO\u001b[0m:     Waiting for application shutdown.\n",
            "\u001b[32mINFO\u001b[0m:     Application shutdown complete.\n",
            "\u001b[32mINFO\u001b[0m:     Finished server process [\u001b[36m10502\u001b[0m]\n",
            "\n",
            "サーバープロセスが終了しました。\n",
            "Task exception was never retrieved\n",
            "future: <Task finished name='Task-1' coro=<Server.serve() done, defined at /usr/local/lib/python3.11/dist-packages/uvicorn/server.py:68> exception=KeyboardInterrupt()>\n",
            "Traceback (most recent call last):\n",
            "  File \"/usr/local/lib/python3.11/dist-packages/uvicorn/main.py\", line 580, in run\n",
            "    server.run()\n",
            "  File \"/usr/local/lib/python3.11/dist-packages/uvicorn/server.py\", line 66, in run\n",
            "    return asyncio.run(self.serve(sockets=sockets))\n",
            "           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
            "  File \"/usr/local/lib/python3.11/dist-packages/nest_asyncio.py\", line 30, in run\n",
            "    return loop.run_until_complete(task)\n",
            "           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n",
            "  File \"/usr/local/lib/python3.11/dist-packages/nest_asyncio.py\", line 92, in run_until_complete\n",
            "    self._run_once()\n",
            "  File \"/usr/local/lib/python3.11/dist-packages/nest_asyncio.py\", line 133, in _run_once\n",
            "    handle._run()\n",
            "  File \"/usr/lib/python3.11/asyncio/events.py\", line 84, in _run\n",
            "    self._context.run(self._callback, *self._args)\n",
            "  File \"/usr/lib/python3.11/asyncio/tasks.py\", line 360, in __wakeup\n",
            "    self.__step()\n",
            "  File \"/usr/lib/python3.11/asyncio/tasks.py\", line 277, in __step\n",
            "    result = coro.send(None)\n",
            "             ^^^^^^^^^^^^^^^\n",
            "  File \"/usr/local/lib/python3.11/dist-packages/uvicorn/server.py\", line 69, in serve\n",
            "    with self.capture_signals():\n",
            "  File \"/usr/lib/python3.11/contextlib.py\", line 144, in __exit__\n",
            "    next(self.gen)\n",
            "  File \"/usr/local/lib/python3.11/dist-packages/uvicorn/server.py\", line 330, in capture_signals\n",
            "    signal.raise_signal(captured_signal)\n",
            "KeyboardInterrupt\n"
          ]
        }
      ],
      "source": [
        "!python app.py"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "RLubjIhbELSR"
      },
      "source": [
        "FastAPIが起動すると、APIとクライアントが通信するためのURL（エンドポイント）が作られます。\n",
        "\n",
        "URLが作られるのと合わせて、Swagger UIというWebインターフェースが作られます。\n",
        "\n",
        "Swagger UIにアクセスすることで、APIの仕様を確認できたり、APIをテストすることができます。\n",
        "\n",
        "Swagger UIを利用することで、APIを通してLLMを動かしてみましょう。"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "XgumW3mGELSR"
      },
      "source": [
        "後片付けとして、使う必要のないngrokのトンネルを削除します。"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 29,
      "metadata": {
        "id": "RJymTZio-WPJ"
      },
      "outputs": [],
      "source": [
        "from pyngrok import ngrok\n",
        "ngrok.kill()"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "T4"
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "GPU"
  },
  "nbformat": 4,
  "nbformat_minor": 0
}