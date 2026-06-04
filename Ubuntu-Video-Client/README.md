# Ubuntu-Video-Client

Ubuntu receiver for the G1 WBCD camera stream.

The client receives the `OrinVideoSender` TCP video format:

```text
[4-byte big-endian payload length][H.264 payload]
```

It is designed for the `main_wbcd_zed_d405.cpp` sender output:

```text
2560x1440@30
top row:    ZED left / ZED right
bottom row: D405 left / D405 right
```

## Install

```bash
sudo apt install -y ffmpeg
```

## Recommended: Orin Direct Send

Start Ubuntu first:

```bash
cd /home/long/workspace_wbcd_sonic/third_party/XRoboToolkit-Orin-Video-Sender/Ubuntu-Video-Client

python3 ubuntu_video_client.py \
  --listen-host 0.0.0.0 \
  --port 12345 \
  --width 2560 \
  --height 1440 \
  --fps 30
```

Then start Orin:

```bash
export LC_ALL=C
export LANG=C

./OrinVideoSender \
  --send \
  --server <Ubuntu_IP> \
  --port 12345 \
  --left-d405-serial 260322271510 \
  --right-d405-serial 260322271351 \
  --bitrate 8000000
```

## Compatible: Orin Listen Mode

Start Orin:

```bash
export LC_ALL=C
export LANG=C

./OrinVideoSender \
  --listen 192.168.123.164:13579 \
  --left-d405-serial 260322271510 \
  --right-d405-serial 260322271351 \
  --bitrate 8000000
```

Start Ubuntu:

```bash
python3 ubuntu_video_client.py \
  --mode request \
  --orin-ip 192.168.123.164 \
  --control-port 13579 \
  --receiver-ip <Ubuntu_IP> \
  --port 12345 \
  --width 2560 \
  --height 1440 \
  --fps 30 \
  --bitrate 8000000
```

## Save Raw H.264

```bash
python3 ubuntu_video_client.py \
  --listen-host 0.0.0.0 \
  --port 12345 \
  --dump out.h264
```

Replay later:

```bash
ffplay -fflags nobuffer -flags low_delay -framedrop -i out.h264
```

## No Display

Use this when you only want receive FPS/bitrate statistics:

```bash
python3 ubuntu_video_client.py --no-display --port 12345
```

## Diagnosis

- Ubuntu also stutters: inspect Orin compose, hardware encoding, TCP send, or network.
- Ubuntu is smooth but PICO stutters: inspect PICO Wi-Fi, PICO decoder/rendering, or Unity config.
- `out.h264` replay stutters: the encoded stream itself is likely uneven before PICO receives it.

If Ubuntu firewall is enabled:

```bash
sudo ufw allow 12345/tcp
```
