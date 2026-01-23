from multiprocessing.connection import Client, Listener
from dataclasses import dataclass
from typing import Optional


# coordinates are normalized (0.0 to 1.0)
@dataclass
class BoundingBox:
    conf: float
    left: float
    top: float
    width: float
    height: float

    def center(self) -> tuple[float, float]:
        cx = self.left + self.width / 2.0
        cy = self.top + self.height / 2.0
        return (cx, cy)

    def get_rotate_90_deg(self):
        return BoundingBox(
            conf=self.conf,
            left=1 - (self.top + self.height),
            top=self.left,
            width=self.height,
            height=self.width,
        )


@dataclass
class CVData:
    pts_s: float
    fps: float
    bounding_box: Optional[BoundingBox]


def create_rocam_ipc_server():
    return Listener(('localhost', 5000))


# Will block until connected to the server
def create_rocam_ipc_client():
    return Client(('localhost', 5000))
