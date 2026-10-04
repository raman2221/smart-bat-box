# FOMO bat detector on Arduino UNO Q

This program runs on the UNO Q's **Linux side**, using Python and a USB webcam.
It is not an Arduino sketch for the microcontroller. Train a FOMO object
detection model in Edge Impulse with one label, exactly `bat`.

The default setup works without a monitor, including over SSH or from Arduino
App Lab. Run the commands below in a Linux terminal on the board.

## 1. Download the model

In Edge Impulse Studio, open **Deployment**, select **Linux (AARCH64)**, build
the deployment, and download the `.eim` file. Copy it to the board, for example
as `model.eim` in your `smart-bat-box` folder. Use the ARM64/AARCH64 deployment
for the UNO Q, rather than an x86 model or an Arduino library.

Alternatively, if the Edge Impulse Linux CLI is already installed on the board,
download your project's model with:

```bash
edge-impulse-linux-runner --download model.eim
```

The CLI asks you to sign in and select your project. See the official
[Edge Impulse Linux setup guide](https://docs.edgeimpulse.com/docs/edge-impulse-for-linux)
if you need to install that CLI. Python only needs the resulting `.eim` file.

Go to the project folder and make the model executable:

```bash
cd ~/smart-bat-box
chmod +x model.eim
```

Replace `~/smart-bat-box` if you copied the project somewhere else. Keep the
`.eim` file on the board; the Python SDK starts it as a local executable.

## 2. Set up Python

Create a virtual environment so these packages stay separate from system Python:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip python3-dev build-essential portaudio19-dev
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r uno_q/requirements.txt
```

Activate the venv again with `source .venv/bin/activate` in each new terminal.
The requirements use `opencv-python-headless` to avoid GUI/libGL dependencies.
The SDK also depends on PyAudio, even for this camera-only program; the build
tools and PortAudio headers above let pip build that dependency on Debian.
No SDK, camera, or model is needed to run the unit tests.

## 3. Find the webcam

Plug the USB webcam into the UNO Q and list its video devices:

```bash
ls /dev/video*
```

Start with `--camera 0` for `/dev/video0`. If that does not work, try
`--camera 1` or `--camera 2`. Some webcams expose multiple video devices, and
only one of them delivers usable video. Close any other program using the camera.

## 4. Run without a window (SSH / App Lab)

From the project folder, with the venv active:

```bash
python uno_q/run_fomo.py --model model.eim --camera 0 --no-window
```

All options together:

```bash
python uno_q/run_fomo.py --model model.eim --camera 0 --threshold 0.6 --no-window --frames-needed 3 --save-dir snapshots
```

Stop with **Ctrl+C**. Headless mode never calls OpenCV window functions. For
App Lab, use this command in the board's Linux terminal, or launch the script
with the same arguments using the app's Python environment after installing
the requirements in that environment.

## 5. Run with a window

Window mode needs a monitor and a working graphical desktop on the UNO Q.
Replace headless OpenCV in your venv; do not keep both OpenCV packages installed:

```bash
python -m pip uninstall -y opencv-python-headless
python -m pip install opencv-python
sudo apt install -y libgl1 libglib2.0-0
python uno_q/run_fomo.py --model model.eim --camera 0
```

The live video shows only large green **BAT DETECTED** or red **NOT A BAT**
text. Press **q** in the video window or **Ctrl+C** to stop. To switch back to
headless OpenCV, uninstall `opencv-python` and install `opencv-python-headless`.

## What the output means

```text
[21:04:13] BAT DETECTED (0.87, 7.9 fps)
Snapshot saved: snapshots/bat_20261004_210413_123456.jpg
[21:04:23] FPS: 7.8
[21:04:25] NOT A BAT (0.00, 7.7 fps)
```

A frame counts as positive when at least one `bat` box has confidence **at or
above** `--threshold`. With the defaults, at least **3 of the last 5 frames**
must be positive to show BAT DETECTED. They need not be consecutive. During
startup, unseen frames count as misses. Several bats in one frame still count
as just one positive frame. Choose `--frames-needed` from 1 to 5 and
`--threshold` from 0 to 1.

The program prints its first status, then prints status again only when it
changes. The score is the strongest qualifying bat confidence in the current
frame (0.00 if none). FPS measures processed camera frames per second, including
inference, rather than the webcam's advertised frame rate. A separate FPS line
appears about every 10 seconds, after the next inference finishes.

Each change to BAT DETECTED saves **one JPEG**, rather than one per frame.
Snapshots show the exact resized/cropped image used by the model, with green
circles at the detected centroids. This keeps the coordinates correct, even
when the camera and model have different aspect ratios. FOMO's small boxes
represent centroid locations, not the full outline of a bat. Snapshot names
include microseconds and gain a counter if a name already exists; existing
photos are never overwritten. Relative paths are relative to your current
terminal folder. This detector does not count entrance IN/OUT events.

## Troubleshooting

- **Model file missing:** check `ls -l model.eim` and the path after `--model`.
- **Permission denied / model not executable:** run `chmod +x model.eim`.
  If it still fails, check that its folder allows execution and that you
  downloaded Linux AARCH64. Also check that the snapshot folder is writable.
- **SDK missing:** activate the venv, then run `python -m pip install edge_impulse_linux`.
- **libGL.so.1 missing:** run `sudo apt install -y libgl1 libglib2.0-0`.
  For SSH / App Lab, prefer the headless package: in your venv, run
  `python -m pip uninstall -y opencv-python`, then
  `python -m pip install --force-reinstall opencv-python-headless`, and add
  `--no-window`. Reinstalling repairs the shared `cv2` files if both packages
  were previously installed.
- **Window cannot open:** use `--no-window` for SSH / App Lab. Local window
  mode needs `opencv-python` and a graphical display; the headless package
  deliberately has no window support.
- **Camera cannot open / frames cannot be read:** try other `--camera` indexes,
  run `ls /dev/video*`, reconnect the webcam, and close apps using it.
  Check `ls -l /dev/video0` and `groups` for access permissions. On Debian,
  if the device belongs to group `video`, an administrator can add your user
  with `sudo usermod -aG video "$USER"`; log out and back in afterward.
- **Slow FPS:** try a smaller **96x96** input when building and retraining the
  impulse, and try **grayscale** in the image processing block. Rebuild and
  download the new `.eim` afterward. Keep other heavy apps closed and use
  `--no-window` to avoid display work. Check accuracy again after changing
  input size or color mode; smaller inputs can miss small bats.
- **Bats are missed:** ensure the camera view and lighting resemble training
  data. Lowering the threshold only accepts boxes the model actually returns;
  it cannot recover boxes filtered out by the model's own detection threshold.

## SDK image handling

The SDK's [image implementation](https://github.com/edgeimpulse/linux-sdk-python/blob/master/edge_impulse_linux/image.py)
and [still-image example](https://github.com/edgeimpulse/linux-sdk-python/blob/master/examples/image/classify-image.py)
expect **RGB** images. The program converts OpenCV's BGR webcam frames before
extracting features, then calls `runner.classify(features)`. It uses
`get_features_from_image_auto_studio_settings` when available to follow Studio's
resize mode, and falls back to `get_features_from_image` on older SDKs. The
fallback uses a center crop, so use matching Studio settings with an older SDK.
Grayscale is converted from camera BGR first, then replicated across channels
to handle SDK versions with different internal gray conversion weights.

## Check the code without hardware

From the project root on Linux, with any Python environment:

```bash
python -m unittest tests.test_run_fomo -v
python -m py_compile uno_q/run_fomo.py
```

The tests use fake frames, OpenCV, and a fake SDK runner. They do not load a
model, open a camera, or require OpenCV, NumPy, or the SDK to be installed.
