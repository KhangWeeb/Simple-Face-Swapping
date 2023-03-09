# USAGE
# tkinter_test.py

# import the necessary packages
from tkinter import *
from tkinter import filedialog

from PIL import Image
from PIL import ImageTk
import cv2
from FaceSwapping import face_swap, scale_swap

def scale(img):
	scale_percent = img.shape[0] / 400
	width = int(img.shape[1] / scale_percent)
	dim = (width, 400)
	# resize image
	resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
	return resized

def select_image1():
	global panelA, imageA
	path = filedialog.askopenfilename()

	if len(path) > 0:
		image = cv2.imread(path)
		imageA = image
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		image = scale(image)
		image = Image.fromarray(image)
		image = ImageTk.PhotoImage(image)
		if panelA is None:
			panelA = Label(image=image)
			panelA.image = image
			panelA.pack(side="left", padx=10, pady=10)

		else:
			panelA.configure(image=image)
			panelA.image = image

def select_image2():
	global panelB, imageB
	path = filedialog.askopenfilename()

	if len(path) > 0:
		image = cv2.imread(path)
		imageB = image
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

		image = scale(image)
		image = Image.fromarray(image)
		image = ImageTk.PhotoImage(image)

		if panelB is None:
			panelB = Label(image=image)
			panelB.image = image
			panelB.pack(side="right", padx=10, pady=10)

		else:
			panelB.configure(image=image)
			panelB.image = image

def scale_result():
	global imageA, imageB, panelC
	result = scale_swap(imageA, imageB)
	result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
	result = scale(result)
	result = Image.fromarray(result)
	result = ImageTk.PhotoImage(result)

	if panelC is None:
		panelC = Label(image=result)
		panelC.image = result
		panelC.pack(side="right", padx=10, pady=10)

	else:
		panelC.configure(image=result)
		panelC.image = result

def result():
	global imageA, imageB, panelC
	result = face_swap(imageA, imageB)
	result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
	result = scale(result)
	result = Image.fromarray(result)
	result = ImageTk.PhotoImage(result)

	if panelC is None:
		panelC = Label(image=result)
		panelC.image = result
		panelC.pack(side="right", padx=10, pady=10)

	else:
		panelC.configure(image=result)
		panelC.image = result
# initialize the window toolkit along with the two image panels
root = Tk()
panelA = None
panelB = None
panelC = None
imageA = None
imageB = None

# create a button, then when pressed, will trigger a file chooser
# dialog and allow the user to select an input image; then add the
# button the GUI
btn1 = Button(root, text="Select an image 1", command=select_image1)
btn1.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn2 = Button(root, text="Select an image 2", command=select_image2)
btn2.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn3 = Button(root, text="Face Swap", command=result)
btn3.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
btn4 = Button(root, text="Face Resize Swap", command=scale_result)
btn4.pack(side="bottom", fill="both", expand="yes", padx="10", pady="10")
# kick off the GUI
root.mainloop()
#Load xong roi do
#h resize