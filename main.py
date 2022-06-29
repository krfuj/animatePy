import gc
import math
from PIL import Image

import torch
import numpy as np 
from torchvision.transforms.functional import to_tensor, center_crop
from encoded_video import Encoded_video, write_video
from Ipython.display import encoded_video


# Loading the model.
print("Big brains loading,  Model loading...")
model = torch.hub.load(
    "AK391/animegan2-pytorch:main",
    "generator",
    pretrained=True,
    device="cpu",
    progress=True,

)


"""
    It takes an image, crops it to a square, resizes it to a given size, and then runs it through the
    model
    
    :param model: The model to use
    :type model: torch.nn.Module
    :param img: The image to be painted
    :type img: Image.Image
    :param size: The size of the image to be generated, defaults to 512
    :type size: int (optional)
    :param device: The device to run the model on. This can be either 'cpu' or 'cuda', defaults to cpu
    :type device: str (optional)
    :return: A tensor of size (3, 512, 512)
    """
def facePaint(model: torch.nn.Module, img: Image.Image, size: int = 512, device: str = 'cpu'):
  
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) //2, (h - s) //2, (w + s) //2, (h + s) //2))
    img = img.resize((size, size), Image.LANCZOS)

    with torch.no_grad():
        input = to_tensor(img).unsqueeze(0) * 2 - 1
        output = model(input.to(device)).cpu()[0]

        output = (output * 0.5 + 0.5).clip(0, 1) * 255.0

    return output


"""
    It takes a tensor X, and subsamples it along the dimension dim, by taking nun_samples samples from
    it
    
    :param X: torch.Tensor - the tensor to be subsampled
    :type X: torch.Tensor
    :param nun_samples: number of samples to subsample to
    :type nun_samples: int
    :param dim: the dimension of the tensor to be subsampled
    :type dim: int
    :return: A tensor with the same shape as X, but with nun_samples elements along the dim dimension.
    """
def iniform_temporal_subsample(X: torch.Tensor, nun_samples: int, dim: int = -3) -> torch.Tensor:
   
    t = x.shape[dim]
    assert nun_samples > 0 and t > 0

    index = torch.linspace(0, t - 1, nun_samples)
    index = torch.clamp(index, 0, t - 1).long()

    return torch.index_select(x, dim, index)



    """
    > Given a 4D tensor, scale the shorter side to a given size, and then scale the longer side to match
    the shorter side
    
    :param x: the input tensor
    :type x: torch.Tensor
    :param size: the size of the shorter side of the image
    :type size: int
    :param interpolation: The interpolation mode to calculate output values. Can be one of "nearest",
    "linear" (3D-only), "bilinear", "bicubic" (4D-only), "trilinear" (5D-only), "area", defaults to
    bilinear
    :type interpolation: str (optional)
    :return: A tensor of shape (c, t, new_h, new_w)
    """

def short_side_scale(x: torch.Tensor, size: int, interpolation: str = "bilinear",) -> torch.Tensor:

    assert len(x.shape) == 4
    assert x.dtype == torch.float32
    c, t, h, w = x.shape
    if w < h:
        new_h = int(math.floor((float(h) / w)* size))
        new_w = size
    else:
        new_h = size
        new_w = int(math.floor((float(w) / h)* size))
    return torch.nn.functional.interpolate(x, size=(new_h, new_w), mode=interpolation, align_corners=False)


    """
    It takes a video, a start time, a duration, and a desired output framerate, and returns a video of
    the same duration but with the desired framerate
    
    :param vid: the video object
    :param start_sec: The starting time of the video clip to be processed
    :param duration: The duration of the clip to be processed
    :param out_fps: The output fps of the video
    """
def inference_step(vid, start_sec, duration, out_fps):
    clip = vid.get_clip(start_sec, start_sec + duration)
    video_arr = torch.from_numpy(clip['video']).permute(3, 0, 1, 2)
    audio_arr = np.expand_dims(clip['audio'], 0)
    audio_fps = None if not vid._has_audio else vid._container.streams.audio[0].sample_rate

    x = uniform_temporal_subsample(video_arr, duration * out_fps)
    x = center_crop(short_side_scale(x, 512), 512)
    x /= 255.0
    x = x.permute(1, 0, 2, 3)
    with torch.no_grad():
        output = model(x.to('cpu')).detach().cpu()
        output = (output * 0.5 + 0.5).clip(0, 1) * 255.0
        output_video = output.permute(0, 2, 3, 1).numpy()

    return output_video. audio_arr, out_fps,audio_fps

def predict_fn(filepath, start_sec, duration):
    """
    It takes a video file, and a start time and duration, and returns a new video file with the
    specified duration starting at the specified start time
    
    :param filepath: the path to the video file
    :param start_sec: The starting second of the video to start processing
    :param duration: The number of seconds to process
    :return: the path to the output video.
    """
    out_fps = 18
    vid = EncodeVideo.from_path(filepath)
    for i in range(duration):
        print(f"Processing step{i + 1} / {duration}...")
        video, audio, fps, audio_fps = inference_step(vid=vid, start_sec = i + start_sec, duration = 1, out_fps = out_fps)
        gc.collect()
        if i ==0:
            video_all = video
            audio_all = audio
        else:
            video_all = np.concantenate((video_all, video))
            audio_all = np.hstack((audio_all, audio))
    print("Writting output video....")

    try:
        write_video('win.mp4',video_all, fps=fps, audio_array=audio_all, audio_fps=audio_fps, audio_codec='acc')
    except:
        print("Error when writing with audio... trying without audio")
        write_video('win.mp4', video_all, fps=fps)

    print(f"Done!")
    del video_all
    del audio_all

    return 'win.mp4'   
