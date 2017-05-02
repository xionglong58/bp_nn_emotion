#! /usr/bin/env python3
# -*-coding: utf-8-*-

import os
import numpy as np
import cv2
import math
from sklearn import decomposition


np.random.seed(0)

def show(img):
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sigmoid(x,derivate=False):
    if derivate is True:
        return x * (1-x)
        # return sigmoid(x) * (1-sigmoid(x))
    return 1/(1 + np.exp(-x))


def relu(x,derivate=False):
    if derivate is True:
        x[x<0] = 0
        x[x>0] = 1
        return x
    x[x<0] = 0
    return x


def tanh(x,derivate=False):
    if derivate is True:
        return 1 - tanh(x)**2
    return np.tanh(x)


CURDIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
DBDIR = os.path.join(CURDIR,'data')
ERROR_LOG_FILE = os.path.join(DBDIR,'error.log')


if not os.path.exists(DBDIR):
    os.makedirs(DBDIR)


class Bpnn(object):
    '''Back propagation neural network implementation 

    Define and implement BP neural network,wrapping common fundamental functions in this class
    like 'predict','train','test' .etc 
    u can use this wrap such like that::
        # create BP neural network 
        >> bp = Bpnn(num_input_layer,num_hidden_layer,num_output_layer)
    '''
    def __init__(self,num_input_layer,num_hidden_layer,num_output_layer,batch=1,learn=0.1):
        np.random.seed(1)
        self.num_input = num_input_layer
        self.num_hidden = num_hidden_layer
        self.num_output = num_output_layer
        self.batch = batch
        self.learn = learn

        self.in_cells = None
        self.hid_cells = None
        self.out_cells = None

        self.in_weights = np.random.normal(0.1,0.01,(num_input_layer,num_hidden_layer))
        self.out_weights = np.random.normal(0.1,0.01,(num_hidden_layer,num_output_layer))

        self.in_bias = None
        self.out_bias = None
        self.train_errors = []
    
    def update(self,train=False):
        if train is True:
            self.in_cells = np.zeros((self.batch,self.num_input))
            self.hid_cells = np.zeros((self.batch,self.num_hidden))
            self.out_cells = np.zeros((self.batch,self.num_output))
            # self.hid_bias = np.random.normal(0.1,0.01,(self.batch,self.num_hidden))
            self.hid_bias = np.zeros((self.batch,self.num_hidden))
            # self.out_bias = np.random.normal(0.1,0.01,(self.batch,self.num_output))
            self.out_bias = np.zeros((self.batch,self.num_output))
        else:
            self.in_cells = np.zeros((1,self.num_input))
            self.hid_cells = np.zeros((1,self.num_hidden))
            self.out_cells = np.zeros((1,self.num_output))
            self.hid_bias = np.random.normal(0.1,0.01,(1,self.num_hidden))
            self.out_bias = np.random.normal(0.1,0.01,(1,self.num_output))

    def predict(self, batch_inputs,train=False):
        if train is False:
            self.update(train)
        # validate train data shape as same as initial in_cells
        assert len(self.in_cells) == len(batch_inputs),"input layer row is not equal to actual input layer"
        assert len(self.in_cells[0]) == len(batch_inputs[0]),"input layer col is not equal to actual input layer"

        # assign input to input layer
        self.in_cells[:,:] = batch_inputs[:,:]

        # activate hidden layer
        layer1 = np.dot(self.in_cells,self.in_weights) #+ self.hid_bias
        self.hid_cells[:,:] = tanh(layer1)

        # activate output layer
        layer2 = np.dot(self.hid_cells,self.out_weights) #+ self.out_bias
        self.out_cells[:,:] = tanh(layer2)

    def back_propagate(self, label):
        # validate label shape as same as output layer
        assert len(label) == len(self.out_cells)
        assert len(label[0] == len(self.out_cells[0]))

        # output layer error
        # layer2_deltas (bat,out)
        layer2_error = label - self.out_cells
        layer2_deltas = layer2_error * tanh(self.out_cells,derivate=True)

        # hidden layer error
        # layer1_deltas (bat,hid)
        layer1_error = np.dot(layer2_deltas,self.out_weights.T)
        layer1_deltas = layer1_error * tanh(self.hid_cells,derivate=True)
        
        # update out weights
        # change (hid,out)
        change = np.dot(self.hid_cells.T,layer2_deltas)
        self.out_weights += self.learn * change 

        # update input weights
        # change (in,hid)
        chage = np.dot(self.in_cells.T,layer1_deltas)
        self.in_weights += self.learn * chage 

        # update bias
        self.out_bias += (self.learn * layer2_deltas)
        self.hid_bias += (self.learn * layer1_deltas)

        # calc one epoch error and return
        error = np.mean(.5*(label - self.out_cells)**2)
        return error

    def train(self, samples, labels, batch, epochs=1000, train=True):
        '''Train BP neural network

        Parameters:
            samples: trainning data collections
            labels: actual correct input
            batch: one batch count
            epochs: epoch of trainning,default 1000
            learn: learning rate,default 0.05
            correct: correct value
        '''
        parts = int(len(samples)/batch)
        print('[^_^] Starting Train...')
        self.update(train)
        for e in range(epochs):
            error = float(0)
            for i in range(parts):
                batch_labels = labels[i:i+batch,:]
                batch_inputs = samples[i:i+batch,:]
                # feed forward
                self.predict(batch_inputs,train)
                error += self.back_propagate(batch_labels)
            error = np.mean(np.abs(error))
            self.train_errors.append(error)
            print('[^_^] Trainning error in %d epoch: %f' % (e,error))
        print('[^_^] Trainning Successfully.')
        # np.savetxt(ERROR_LOG_FILE,self.train_errors,fmt='%.5f')
        # print('[^_^] Train error list save in %s successfully.' % ERROR_LOG_FILE)

    @staticmethod
    def extract(lstfile,num_label,encode='utf8'):
        '''Extract lst file image and create output label

        1. happy    -->     [1, 0, 0, 0, 0, 0, 0]
        2. sad      -->     [0, 1, 0, 0, 0, 0, 0]
        3. surprise -->     [0, 0, 1, 0, 0, 0, 0]
        4. angry    -->     [0, 0, 0, 1, 0, 0, 0]
        5. disgust  -->     [0, 0, 0, 0, 1, 0, 0]
        6. fear     -->     [0, 0, 0, 0, 0, 1, 0]
        7. normal   -->     [0, 0, 0, 0, 0, 0, 1]
        '''
        label_mapping = dict()
        for i in range(num_label):
            label = np.zeros(num_label,dtype=np.uint8)
            index = num_label-1-i
            label[index] = 1
            label_mapping[index] = list(label)
        # print(label_mapping)

        data = []
        if not os.path.exists(lstfile):
            raise Exception("Nothing!! (extract/bp.py)")
        idx_label_file = np.load(lstfile)['label_file']
        for lf in idx_label_file:
            idx,label,file = lf
            idx = int(idx)
            label = label_mapping[int(label)]
            img = cv2.imread(file,cv2.IMREAD_UNCHANGED)
            img = Bpnn.pre_deal_image(img)
            data.append(([idx],np.array(label),np.array(img)))
        return np.array(data)
    
    @staticmethod
    def minmax(data):
        return (data - np.min(data))/(np.max(data) - np.min(data)+1)
    
    @staticmethod
    def reduceDim(dataMat,n=800):  
        meanVal = np.mean(dataMat)
        meanMat = dataMat - meanVal
        t = meanMat * meanMat.T
        eigval,eigvec = np.linalg.eig(t)
        return lowDDataMat,reconMat 
        
    @staticmethod
    def trunck(data,chunk=10):
        rows,_ = data.shape
        if rows % chunk == 0:
            return data
        else:
            mod = rows % chunk
            return data[:-mod,:]
        
    @staticmethod
    def split_train_test(data,test_parts=30):
        trains = data[:-test_parts,:]
        tests = data[-test_parts:,:]
        print('[^_^] Split Train data and Test data sunccessfully~')
        return trains,tests

    @staticmethod
    def split_label_data(data):
        assert len(data) != 0,"A empty array is given"
        assert  len(data[0]) == 3,"Need 3 cols array."
        idx = [ele[0] for ele in data[:,0]]
        labels = data[:,1] 
        dat = data[:,2]
        print('[^_^] Split labels and data sunccessfully~')
        return idx,np.array(list(labels),dtype=np.float32),np.array(list(dat))
    
    @staticmethod
    def getInitialParams(data,num_output=7,num_hidden=None):
        assert len(data.shape) == 2,"Need shape 2-d array."
        _,num_input =  data.shape
        if num_hidden is None:
            if int(num_input/num_output) > 1000:
                num_hidden = int((num_input+num_output)/3*2)
            else:
                num_hidden = num_input/2
        elif num_hidden<30:
            num_hidden = int(np.sqrt(num_input+num_output) + num_hidden)
        return num_input,num_hidden,num_output
    
    @staticmethod
    def binaryzation(img):
        kernel = np.ones((2,2),np.uint8)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.blur(img,(1,1))
        img = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel,iterations=2)
        img = cv2.threshold(img,70,255,cv2.THRESH_BINARY+cv2.THRESH_TRUNC)[1]
        # thresh = np.mean(img)
        # img[img>thresh] += 100
        img = cv2.equalizeHist(img)
        # print(img);exit()
        return img
    
    @staticmethod
    def dct(img):
        assert hasattr(img,"shape"),"Need a numpy array instance"
        assert len(img.shape) == 2,"img must be a gray image"
        # img = cv2.medianBlur(img,3)
        data = np.array(img,np.float)/255   
        trans = cv2.dct(data)
        dct1 = np.uint8(trans)*255
        lwf = dct1[:8,:8]
        return lwf
    
    @staticmethod
    def pre_deal_image(img):
        assert isinstance(img,np.ndarray),'Img must be a cv2.imread instance'
        if len(img.shape) == 3:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(92,112),interpolation=cv2.INTER_CUBIC)
        # img = Bpnn.binaryzation(img)
        img = Bpnn.dct(img)
        img = Bpnn.minmax(img)
        img = img.flatten()
        return img

    @staticmethod
    def extract_origin():
        srcfile = os.path.join(CURDIR,'trains','jaffe.csv')
        imgdir = os.path.join(CURDIR,'trains','jaffe')
        lbls = np.loadtxt(srcfile,bytes,delimiter=',',usecols=(0,1,2,3,4,5,6,7))
        data = []
        for lbl in lbls:
            idx = int(lbl[0])
            labels = Bpnn.minmax(np.array(lbl[1:-1],np.float))
            filename = lbl[-1].decode('utf8').replace('-','.')
            filename = '.'.join([filename,str(idx),'tiff'])
            filename = os.path.join(imgdir,filename)
            # print(labels)
            # exit()
            if os.path.exists(filename):
                img = cv2.imread(filename,cv2.IMREAD_UNCHANGED)
                # _show(img)
                img = Bpnn.pre_deal_image(img)
                data.append(([idx],labels,img))
        return np.array(data)

    @staticmethod
    def test():
        bp = Bpnn(4,8,2,4)
        samples = np.array([[0,0,1,1],[1,1,0,0],[1,0,1,0],[1,1,1,1]])
        print(samples.shape)
        label = np.array([[0,1], [1,0], [0,0], [1,1]])
        bp.train(samples,label,4,30000)
        print(label)
        print(bp.out_cells)
        exit()


def pre_deal_image(img):
    assert isinstance(img,np.ndarray),'Img must be a cv2.imread instance'
    if len(img.shape) == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(92,112),interpolation=cv2.INTER_CUBIC)
    # img = Bpnn.binaryzation(img)
    img = Bpnn.dct(img)
    img = Bpnn.minmax(img)
    test = img.flatten()
    h,w = img.shape
    img.shape = (1,h*w)
    return img


def _show(img):
    cv2.imshow('show',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == '__main__':
    # 1-->happy   2-->sadness   3-->surprise
    # 4-->angry   5-->disgust   6-->fear
    # 7-->normal
    # Bpnn.test()
    import sys
    num_hidden = int(sys.argv[1])
    lst = os.path.join(CURDIR,'train.lst.npz')
    batch = 10
    # data = Bpnn.extract(lst,7)
    data = Bpnn.extract_origin()
    data = Bpnn.trunck(data)
    # traindata,testdata = Bpnn.split_train_test(data)
    idx,train_labels,train_data = Bpnn.split_label_data(data)
    # print(train_data.shape)
    idx,test_labels,test_data = Bpnn.split_label_data(data)
    # print(list(zip(idx,train_labels)))
    # exit()
    
    # num_input_layer,num_hidden_layer,num_output_layer = Bpnn.getInitialParams(data=train_data,num_hidden=19,num_output=6)
    num_input_layer,num_hidden_layer,num_output_layer = Bpnn.getInitialParams(data=train_data,num_hidden=num_hidden,num_output=6)
    # print(num_output_layer)
    print("num_hidden: ",num_hidden_layer)
    input('****************************')
    np.set_printoptions(threshold=5000)
    # print(test_data.shape)
    bnn = Bpnn( # inital BP neural network
            num_input_layer = num_input_layer,
            num_hidden_layer = num_hidden_layer,
            num_output_layer = num_output_layer,
            batch = batch)

    bnn.train(train_data,train_labels,batch=batch,epochs=10000)
    # exit()

    print('\n[^_^] Test Starting\n')
    testdir = os.path.join(CURDIR,'trains','test')
    n = 100;acc = float(0)
    for i in np.random.randint(0,200,(n,)):
        test = np.array([test_data[i]])
        label = test_labels[i]
        # print(test.shape)
        bnn.predict(test)
        # print(idx[i])
        if np.argmax(label)==np.argmax(bnn.out_cells[0]):
            acc += 1
        # print(np.argmax(label))
        print(bnn.out_cells)
        print(label)
        # print(np.argmax(bnn.out_cells[0]))
        print('\n')
        # input()
    print('ending...')
    print(type(n))
    print('acc: %f' % (acc/n))

    # for path,dirs,files in os.walk(testdir):
    #     for imgfile in files:
    #         imgfile = os.path.join(path,imgfile)
    #         # print(imgfile)
    #         img = cv2.imread(imgfile)
    #         img = pre_deal_image(img)
    #         bnn.predict(img)
    #         print(imgfile)
    #         print(np.argmax(bnn.out_cells[0]),":",bnn.out_cells)
    #         input('continue...')
    