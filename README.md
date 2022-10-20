# cglut
### Create your own color-grading 3D-LUT preset using a custom python script.

Huge improvement over [DomBito/muhcolors](https://github.com/DomBito/muhcolors).

### What you can do with it
+ White-balance by just inputing RGB values for which you want the saturation to be zero.
+ RGB curves function that accepts [GIMP](https://github.com/GNOME/gimp) preset files (in the old format) as input.
+ 1-D Curve function, but with extra channel options such as luminance, lightness, chroma and even hue.
+ Powerful "perturb" function, which is perturbation of the identity map of one channel, where the perturbation added is a function of another given channel.
+ Tweak function similar to [AviSynth's Tweak](https://www.avisynth.org.ru/docs/english/corefilters/tweak.htm).
+ Use of visually uniform and linear color-spaces, such as XYZ, xYy, cieLUV, cieLch and hsluv. For that, most of the [hsluv](https://github.com/hsluv/hsluv) project conversions are re-implemented in [NumPy](https://github.com/numpy/numpy/), so it's optimized for arrays.

More details on how the functions work and how to use them in the future.

### Usage:
```
usage: cglut [-h] [-s LSIZE] [-o LUTFILE] [-c] script_file

Script-based color-grading 3D LUT creator.

positional arguments:
  script_file           script file with the color-grading functions

options:
  -h, --help            show this help message and exit
  -s LSIZE, --lut_size LSIZE
                        set LUT size (32, 64, 128, or 256)
  -o LUTFILE, --output LUTFILE
                        Name of the lut file.
  -c, --clean           Doesn't save the preLUT file (preLUT makes consecutive exports faster).
```
Running `cglut` on a script will export a file named `lut.cube`.

### Example of a custom script:
```python
#your_script_file.py
gray_points= [[10,11,6], [150,144,149], [235,215,244]]
balance(gray_points)
rgbcurve(red=[0,0,100,70],green=[0,0,100,100],blue=[0,20,100,90], mode='luminance',strength=0.9)
tweak(hue=10, l_range=[30,80], c_range[50,100], h_range=[180,210], smooth=30)
```

What this example script does:
 - Applies white-balance by shifting the chroma channels along the luminosity so the `gray_points` turns into actual grays (zero saturation).
 - Change the scale of each Red, Green, and Blue channels using curve-like function (with linear interpolation). In this instance, it only affects the luminosity (`mode='luminance'`) and it takes 90% of effect (`strength=0.9`) of what this particular RGB curve would normally do.
 - Adds 10 deg of hue for the colors within the given lightness, chroma and hue ranges. Outside this range, the strength of the filter linearely drops to 0 when the colors are 30 numbers alway from the ranges in each channel (`smooth=30`).

### Using the exported LUT:
You can import the `.cube` file in any program that accepts this format.
However, this is a project made with [vapoursynth](https://github.com/vapoursynth/vapoursynth) in mind. You can import the LUT file using [sekrit-twc/timecube](https://github.com/sekrit-twc/timecube) as follows:
```python
# In this example, src is a YUV420P8, rec709 videonode.
dst = core.resize.Bicubic(clip=src, format=vs.RGBS)
dst = core.timecube.Cube(dst,cube="lut.cube")
dst = core.resize.Bicubic(clip=dst, format=vs.YUV420P8, matrix_s='709')
```
#### Using ffmpeg to apply the LUT:
To get an image or video with your LUT applied to it with[FFmpeg](https://github.com/FFmpeg/FFmpeg), you just need to run
```
ffmpeg -y -v quiet -i VIDEO_OR_IMAGE_INPUT -vf lut3d="CUBE_FILE" VIDEO_OR_IMAGE_INPUT
```
#### Preview with MPV:
You can quickly preview the LUT file on a image or video using [mpv](https://github.com/mpv-player/mpv):
```bash
mpv --keep-open --vf="lavfi=[lut3d=CUBE_FILE]" VIDEO_OR_IMAGE_FILE
```
#### Applying it with Imagemagick
For images only, you can also use [ImageMagick](https://github.com/ImageMagick/ImageMagick):
```bash
magick IMG_INPUT cube:CUBE_FILE -hald-clut IMG_OUTPUT
```
