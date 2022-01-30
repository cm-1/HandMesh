import cv2

import runnerCreator

class webcamManager:
    def __init__(self, runner, width, height):
        self.runner = runner
        cap = cv2.VideoCapture(0) #,cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap = cap

    def displayFrame(self):
        ret, frame = self.cap.read()

        convertedFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not ret:
            return

        handInfo = self.runner.processFrame(convertedFrame)
        colourList = [(0, 0, 255), (0, 255, 0)]
        # Iterate over all hands detected
        for handNum in range(handInfo.points2D.shape[0]):
            
            for landmarkNum in range(handInfo.points2D.shape[1]):
                pt = (handInfo.points2D[handNum][landmarkNum]).astype(int)

                circleRadius = 5
                colour = colourList[handNum]
                thickness = 1
                
                frame = cv2.circle(frame, pt, circleRadius, colour, thickness)

        cv2.imshow("test", frame)



    

print("Starting")
w = 640
h = 480
focalLength = 634
r = runnerCreator.getRunner(focalLength, w)
wcm = webcamManager(r, w, h)
while True:
    wcm.displayFrame()
    # Break on press of "Q" key.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print("Done")

