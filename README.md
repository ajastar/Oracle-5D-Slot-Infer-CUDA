# Island Profiler: The Oracle 5D (CUDA Edition)

**~ パチスロ島シミュレーション & リアルタイム設定判別システム ~**

![Version](https://img.shields.io/badge/version-7.0.0_CUDA-green)
![Tech](https://img.shields.io/badge/Tech-CUDA_C++_Python-blue)
![Platform](https://img.shields.io/badge/Target-RTX_5060_Ti-red)

## 概要 (Overview)

**Island Profiler (Oracle 5D)** は、パチスロ「ジャグラー」シリーズ全機種に対応した、超高速モンテカルロ・シミュレーター兼・設定判別ツールです。

GPU (NVIDIA GeForce RTX 5060 Ti) の並列演算能力（4352 CUDA Cores）をフル活用し、**「3000億ゲーム（60兆試行）」** 規模のシミュレーションデータを事前に生成。
その膨大なビッグデータ（Oracle DB）と照らし合わせることで、**「現在の台データ」だけから「設定期待値」だけでなく「島全体の状況（ベース設定など）」までを5次元的に逆算**します。

### 主な機能
1.  **超高速シミュレーション**: CUDAによる並列処理で、数千時間の稼働データを数秒で生成。
2.  **5次元判別 (5D Profiling)**:
    *   単なるボーナス確率だけでなく、「島全体の平均合算」「ワースト台の挙動」「稼働ゲーム数」など5つのパラメータで照合。
    *   「全台系イベント」や「特定日のベースアップ」などの状況判断も自動化。
3.  **モバイル対応**: PCでサーバーを起動し、スマホからQRコード/URLでアクセス可能。ホール内でのリアルタイム判別を実現。

## 動作環境 (Requirements)

*   **OS**: Windows 10 / 11
*   **GPU**: NVIDIA GeForce RTX 3060以上推奨 (RTX 5060 Ti 最適化済)
*   **Runtime**:
    *   Python 3.10+ (Flask, NumPy)
    *   CUDA Toolkit 12.x
    *   PowerShell 7+

## 使い方 (Usage)

### 1. 起動 (Launch)
フォルダ内の **`start.bat`** をダブルクリックしてください。
以下のシステムが全自動で立ち上がります。
*   **Oracle Server**: 5Dデータベースサーバー (Port 5001)
*   **Ngrok Tunnel**: 外部アクセス用のセキュアトンネル

### 2. スマホでアクセス
起動時に表示される `https://xxxx.ngrok-free.app` というURLをスマホで開いてください。

### 3. データ生成 (Optional)
もしデータベース(`*.bin`)を再生成したい場合は、以下のコマンドを実行してください。
```powershell
.\run_cuda_simulation.ps1
```

## 対応機種 (Supported Machines)
*   マイジャグラーV (My Juggler V)
*   アイムジャグラーEX (I'm Juggler EX)
*   ファンキージャグラー2 (Funky Juggler 2)
*   ハッピージャグラーVIII (Happy Juggler VIII)
*   ゴーゴージャグラー3 (Gogo Juggler 3)
*   ミスタージャグラー (Mr. Juggler)
*   ウルトラミラクルジャグラー (Ultra Miracle Juggler)

## 技術スタック (Tech Stack)
*   **Core Engine**: C++20 / CUDA (Compute Capability 8.9)
*   **Backend**: Python (Flask) / Memory Mapped Files
*   **Frontend**: HTML5 / CSS3 (Dark/Neon Future UI) / Chart.js
*   **Automation**: PowerShell / Batch

---
*Created by Antigravity (Google DeepMind Team)*
