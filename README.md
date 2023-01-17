# Gemma - Sign Recognition through Gemini and MediaPipe
_A quick description and set-up tutorial._

## Description
Inspired by https://github.com/gabguerin/Sign-Language-Recognition--MediaPipe-DTW, GemMa is an expansion that recognizes gestures through their angle composition over video; instead of a single camera, a Ultraleap 3Di camera with custom tracking software (Gemini) is used besides a MediaPipe camera.
The dual camera set-up brings an outlier reduction, as an obscured hand in one camera is still visible by the other. The software runs in Python 3.7, other versions might not work.

## Set-Up
Gemini is only supported in C, one way to still work with the software is by making a custom library and access it through c-types, the next steps will explain (hopefully) everything.

### Azure Kinect DK

The main camera works with an Azure Kinect DK, you can ignore this part of the set-up if you want to work with a normal camera. Otherwise just follow the steps in https://github.com/ibaiGorordo/pyKinectAzure, there are plenty of functionalities that you could experiment with and further integrate into this project.

### Making your own custom Gemini library

_A custom library is already included, but only returns part of the entire possible data (the 21 landmarks of each hand). If you want more data from Gemini, you need to make your own library._

1. Install Gemini from Ultraleap (in Windows!).
2. Follow Ultraleap's README.md to make an environment through cmake (usually found in C:\Program Files\Ultraleap\LeapSDK), I recommend working within their samples page.
3. Edit your CMakeLists.txt file to make a library (this step can be skipped if you use the included CMakeLists.txt in the /assets folder):
    1. Each #include which uses Gemini needs to be added as a library and then this library needs to be linked with the Gemini library.
    2. For the header of the library file, add __declspec(dllexport) to each method that you would want to use in Python (void loadLandmarks() would become __declspec(dllexport) void loadLandmarks()).
4. Run the build, then take the .dll file from the build folder and place it into this work environment *THE BUILD WILL NOT RUN WITHOUT THE PRESENCE OF THE LEAPC.DLL FILE*.
5. Use ctypes in Python to run the c methods and take their data (see leap_listener.py for its usage).

### Running the software
Most computers are unable to quickly run through hundreds of files for quick gesture recognition, because of this the data set has been split in categories that each test similar traits, see the PDF to see what each category tests.

#### Requirements
As previously mentioned, the software runs in Python 3.7, besides this there are a bunch of other imports that are used. To install of these, just run:

`pip install -r requirements.txt`
#### Creating a data set
Creating your own category and data set is done by running sign_recorder.py in the following way:

`py -3.7 recorder.py $CATEGORY $GESTURE`

Every MMB will record your gesture and save the landmarks (3D points on the hand, usually 4 points on each finger and 1 on the palm) recorded by Gemini and the MP4 files which are recorded by the main camera.

The architecture of the data is saved in the following way:
```
|data/
    |-videos/
        |-category1/
            |-Gesture1/
             |-Gesture1-1.mp4
             |-Gesture1-2.mp4
             |-Gesture1-3.mp4
            ...
            |-Gesture2/
          |-category2/
            |-Gesture1/
             |-Gesture1-1.mp4
             |-Gesture1-2.mp4
             ...
            ...
    |-ultraleapdataset/
         |-category1/
            |-Gesture1/
                |-Gesture1-1/
                 |lh_Gesture1-1.pickle
                 |rh-Gesture1-1.pickle
                |-Gesture1-2/
                 |lh_Gesture1-2.pickle
                 |rh-Gesture1-2.pickle
                ...
            |-Gesture2/
                |-Gesture2-1/
                 |lh_Gesture2-1.pickle
                 |rh-Gesture2-1.pickle
                ...
            ...
```

After running the main program, all of the .mp4 files will be converted to their landmark data and saved in the data/dataset folder.

#### Gesture recognition
To run Gemmma do:

`py -3.7 main.py $CATEGORY`

The program will open the catagory folder and load all of the landmarks within this catagory. After everything is loaded in, pressing the MMB will start the cameras and film for the desired amount of frames. When all the 21 landmarks over all the frames are collected, all of the landmarks will get the dot product calculated with each other landmark. After this, the dot product is normalized and the arccos is calculated, which all results in a list of 441 angles (21*21). By comparing the recorded angles with the angles from the dataset through (fast-)DTW, the right gesture can be determined. For further information, view the provided PDF in this project.

## Shortcomings and things to keep in mind
- Since angles of the hands are taken, positioning and rotation of the hand is not taken into consideration, this could be resolved by using Gemini's rotational data as the rotation of the hand is very important data for what the content of a gesture.
- The outlier amount looks bigger than it actually is, this is due to the fact that filming duration is set to one time and gestures can be done before that time limit is reached. The implementation of a button press that finishes the recording would allow for less false negatives.
- For single-handed categories, you can only have one hand in field. This limitation is done so that a hand doing nothing does not influence the result of the actual gesturing hand. A possible way of resolving this would be to look at the movement being done by each hand (one problem with this method however is that the supporting hand sometime doesn't move and supports the gesture done by the dominant hand.)
- Since the main signer was left-handed, the provided data-set is left-handed.
- Currently an old artifact in the program is the fact that a new (and expensive) Azure Camera is only being used for filming. There are plenty of gadgets the camera has (the first thing that comes in mind for this project is the depth camera). Alternatively, the program could just work with regular cameras instead, as this would allow for multi-perspective work at a much cheaper price. All that needs to be rewritten is how the frames are delivered, as MediaPipe works over each provided frame.

