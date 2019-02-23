import cv2, time, pandas
from datetime import datetime

# Create a reference image
first_frame = None
status_list =[None,None]
times = []
df = pandas.DataFrame(columns = ["Start","End"])
# Create a dataframe with two columns Start and End

video = cv2.VideoCapture(0)
while True:
    check, frame = video.read() #check is boolean, frame is ndarray
    # Captures the frame

    status = 0 # no motion flag


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert into gray
    gray = cv2.GaussianBlur(gray,(21,21),0) # Removes noise and improves accuracy of the difference
    # Blur the image

    # Define the 1st frame. i.e. the static image or the reference image to detect motion
    if first_frame is None :
        first_frame = gray
        continue

    delta_frame = cv2.absdiff(first_frame,gray) # find the difference
    # gives another image

    # assign black pixel if no difference and white pixel if difference >30
    thresh_frame = cv2.threshold(delta_frame,30, 255, cv2.THRESH_BINARY)[1]
    # threshold method parameters: image, threshold limit, assignment to threshold pixel, threshold methods

    #smoothing the threshold Frame ,remove black holes
    thresh_frame = cv2.dilate(thresh_frame, None, iterations = 2)
    # parameters: threshold frame, kernel array, no of times to iterate to smooth

    #finding contours
    # find contour or draw contour methods
    (_,cnts,_) = cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    # store contour in a tupule
    # parameters : frames whose contour is found, method to draw to an image, approximation method to retrieve the contour

    # Filter contours
    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
        #area of object you want to detect
            continue

        status = 1 # when python find contour of given size change the status from 0 to 1
        # if countour >10000 pixel


        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)

    status_list.append(status)

    if status_list[-1] == 1 and status_list[-2] == 0: #last and second last item on the list
        times.append(datetime.now())

    if status_list[-1] == 0 and status_list[-2] == 1: #last and second last item on the list
        times.append(datetime.now())

    cv2.imshow("Gray Frame", gray)
    cv2.imshow("Delta Frame",delta_frame)
    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Color Frame", frame )

    key = cv2.waitKey(1) #Make capture rate 1 millisecond

    # print(gray)
    # print(delta_frame)

    if key == ord('q'):
        #if you press the q button it breaks the while loop
        if status == 1:
            times.append(datetime.now())
            #append time at the time when the window is closed/quit
        break
    #print (status)

print(status_list)
print(times)

for i in range(0,len(times),2):
    df = df.append({"Start": times[i],"End":times[i+1]},ignore_index = True)
    # if 6 item, 1st and 2nd item in 1st row, 3rd and 4th item in 2nd row
    # so iterate 6/2 times i.e. 3 times.
    # so iterate with the step of two specified in range(start index, end index, step)

df.to_csv("Times.csv")

video.release() # release the video
cv2.destroyAllWindows  # destroy the actual window seen on the screen
