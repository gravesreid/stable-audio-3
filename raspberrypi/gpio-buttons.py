import subprocess, time, socket, json
from gpiozero import Button, Device
from gpiozero.pins.lgpio import LGPIOFactory
from signal import pause as wait
Device.pin_factory = LGPIOFactory()

SOCK = "/tmp/mpvsocket"
COOLDOWN = 0.75
_last = {}

def mpv(command):
    #send a command to mpv
    print(f"received command {command}")
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(2)
            s.connect(SOCK)
            s.sendall((json.dumps({"command": command}) + "\n").encode())
            return json.loads(s.recv(4096).decode().splitlines()[0])
    except (OSError, ValueError) as e:
        print(f"mpv {command}: {e}", flush=True)
        return None

def script(path):
    print(f"running {path}")

def debounced(name,fn):
    now = time.monotonic()
    if now - _last.get(name,0) < COOLDOWN:
        return
    _last[name] = now
    fn()

btn_pause = Button(5, pull_up=True, bounce_time=0.05)
btn_next = Button(6, pull_up=True, bounce_time=0.05)

btn_pause.when_pressed = lambda: debounced("pause", lambda: mpv(["cycle", "pause"]))
btn_next.when_pressed = lambda: debounced("skip", lambda: mpv(["playlist-next", "force"]))

wait()
                                           
  
