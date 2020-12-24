# Realtime two view registration

## Requirements
* [NumPy]
* [OpenCV] - 4.2.0
* [PyRealsense2]
* [Open3D] - 0.9.0.0


## Usage:
you have to take 2 steps.

first, you take pictures of chessboard and save it.
 
You need calibrate two camera first with a doubleside chessboard.

```
python save_image.py
```

press 's' to save several pictures that ChessBoard appears in both cameras's view. the picture will be saved in the 'output/' directory.

then change line 15 `chessBoard_num = your picture num` in the 'Realtime.py'file.

later 
```
python Realtime.py
```
you will see the registration result.
