# Solving handwritten definite integrals
The goal of this project is to test AIs capability of converting mathematical equations from handwritten form to LaTeX
and using the results to calculate solutions for mathematical problems. The mathematical problems chosen for testing are definite integrals.
Main function of the repository is the translation of images like this:
![image](https://user-images.githubusercontent.com/118485347/202576391-bba06888-e77d-47bf-af47-46cbfcb7ba6a.png)
Into LaTeX formula:
```
\int_{26}^{42}x^5+7x^3+5xdx
```
 The project consists of 3 main functions:
 + Convolutional Neural Network
 + Sequence to Sequence model
 + Numerical Integration methods
 
 After launching the main.ipynb file, writing a definite integral and clicking start, the program will save an image of the equation. Every symbol is detected and saved separatly for later usage.
 ## Convolutional Neural Network
 Dataset used for training:
 www.kaggle.com/xainano/handwrittenmathsymbols
 
 The neural network is trained for image recognition of mathematical symbols. It is taking individual symbols detected earlier and translating them into LaTeX. The accuracy is reaching over 90%.
 ##Sequence to Sequence
 The CNN model is able to recognise individual symbols, but it is also important to know their position because of indexing. The Seq2Seq model was created specifically to solve this problem. Training Seq2Seq models is heavely limited by hardware. The accuracy with given limitations reached approximately 54%.
 The model was trained on a set of equation images generated by a script wich used the aforementioned dataset.
## Mathematical methods
 The converted equations is run by 3 numerical methods:
 + Rectangle Method
 + Trapezoidal Method
 + Simpsons Rule
 
## All three results and the equation are displayed on th screen.
![image](https://user-images.githubusercontent.com/118485347/202584537-3431938c-1d69-464b-a4df-3f53132aa9de.png)
