from typing import Optional, List, Dict

try:
    from .configuration_transunet import TransUNetConfig
except ImportError:
    from configuration_transunet import TransUNetConfig


class TransUNetWithTracksConfig(TransUNetConfig):
    model_type = "transunet_with_tracks"

    def __init__(
        self,
        # Adapter configuration
        num_human_tracks_1bp: int = 1697,
        num_mouse_tracks_1bp: int = 630,
        num_human_tracks_128bp: int = 2733,
        num_mouse_tracks_128bp: int = 310,
        use_trainable_scale: bool = True,
        crop_size: int = 0,
        # Track type arrays stored as lists for JSON serialization
        human_track_types_1bp: Optional[List[str]] = None,
        mouse_track_types_1bp: Optional[List[str]] = None,
        human_track_types_128bp: Optional[List[str]] = None,
        mouse_track_types_128bp: Optional[List[str]] = None,
        **kwargs,
    ):
        self.num_human_tracks_1bp = num_human_tracks_1bp
        self.num_mouse_tracks_1bp = num_mouse_tracks_1bp
        self.num_human_tracks_128bp = num_human_tracks_128bp
        self.num_mouse_tracks_128bp = num_mouse_tracks_128bp
        self.use_trainable_scale = use_trainable_scale
        self.crop_size = crop_size
        self.human_track_types_1bp = human_track_types_1bp
        self.mouse_track_types_1bp = mouse_track_types_1bp
        self.human_track_types_128bp = human_track_types_128bp
        self.mouse_track_types_128bp = mouse_track_types_128bp
        super().__init__(**kwargs)
