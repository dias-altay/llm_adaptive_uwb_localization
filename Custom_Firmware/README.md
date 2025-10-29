# Custom DWM1001 Firmware

## Overview

This folder contains custom firmware for DWM1001 modules, based on the open-source project from [TIERS/dwm1001-uwb-firmware](https://github.com/TIERS/dwm1001-uwb-firmware/).

* **1_Tag_Initiator**: Contains the modified firmware for the Tag (Initiator).

* **2_Anchors_Responder**: Contains the original TIERS firmware for the Anchors (Responders).

The build and configuration instructions below apply to both roles.

## Prerequisites

* **SEGGER Embedded Studio V8.24** (Project was written and tested on this version).

* [Decawave/dwm1001-examples](https://github.com/Decawave/dwm1001-examples) repository.

* DWM1001 module(s).

## SEGGER Embedded Studio Setup

Before building, ensure your SES environment is configured correctly:

1. **Install Packages**: In SES, install the following packages:

   * `CMSIS 5 CMSIS-CORE Support Package`

   * `CMSIS-CORE Support Package`

   * `Nordic Semiconductor nRF CPU Support Package`

2. **Project Options (RTT)**:

   * Go to `Project -> Options -> Code`.

   * Set `Library -> Library I/O` to **RTT**.

3. **Project Options (Linker)**:

   * Go to `Project -> Options -> Linker`.

   * Set `Additional Output Format` to **hex**.

4. **Exclude Files**:

   * Right-click on `SEGGER_RTT_Syscalls_KEIL.c` and `retarget.c` in the project explorer.

   * Select `Exclude from Build`.

## Build Instructions

1. **Clone Base Repository**: Clone the official Decawave examples repository:

```

git clone [https://github.com/Decawave/dwm1001-examples](https://github.com/Decawave/dwm1001-examples)

```

2. **Copy Firmware Files**:

* Navigate to the cloned directory: `dwm1001-examples/examples/ss_twr_init/`.

* Copy all files from either the `1_Tag_Initiator` or `2_Anchors_Responder` folder (from this project).

* Paste and replace the files inside the `ss_twr_init` directory.

3. **Open Project**:

* Navigate to `dwm1001-examples/examples/ss_twr_init/SES/`.

* Open the `ss_twr_init.emProject` file in SEGGER Embedded Studio.

4. **Build Project**:

* In SEGGER Embedded Studio, select `Build -> Build ss_twr_init`.

5. **Download to Module**:

* Connect your DWM1001 module via USB.

* In SEGGER Embedded Studio, select `Target -> Download ss_twr_init`.

## Configuration (Mandatory)

Before flashing the **Tag (Initiator)**, you **must** configure its settings:

1. Open the `ss_init_main.c` file (e.g., in `dwm1001-examples/examples/ss_twr_init/ss_init_main.c`).

2. Set the unique device ID:

* `int my_id = 0;` (This **must be 0** for the Tag).

3. Set the total number of anchors in your network:

* `#define NUM_ANCHORS 6` (Change `6` to match your setup).

Before flashing an **Anchor (Responder)**, you **must** configure its settings:

1. Open the `ss_init_main.c` file.

2. Set the unique device ID (must be non-zero):

* `int my_id = 1;` (Use `1` for the first anchor, `2` for the second, and so on).
