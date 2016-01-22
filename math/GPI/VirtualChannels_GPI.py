# Author: Mike Schar 
# Date: 2015aug27

import gpi
import numpy as np
from gpi import QtCore, QtGui

class ExternalNode(gpi.NodeAPI):
    """ Array compression for virtual channels
        Reduce the number of channels to a user defined number based on the publication by Buehrer Martin et al, MRM 2007
    """

    def initUI(self):
        # Widgets
        self.addWidget('DisplayBox', 'image', interp=True, ann_box='Pointer')
        self.addWidget('Slider', 'image ceiling',val=60)
        self.addWidget('Slider', 'crop left')
        self.addWidget('Slider', 'crop right')
        self.addWidget('Slider', 'crop top')
        self.addWidget('Slider', 'crop bottom')
        self.addWidget('PushButton', 'compute', toggle=True)
        self.addWidget('SpinBox', 'virtual channels', immediate=True, val=12)
        

        # IO Ports
        self.addInPort('data', 'NPYarray', dtype=[np.complex64, np.complex128], obligation=gpi.REQUIRED)
        self.addInPort('noise', 'NPYarray', dtype=[np.complex64, np.complex128], obligation=gpi.REQUIRED)
        self.addInPort('sensitivity map', 'NPYarray', dtype=[np.complex64, np.complex128], obligation=gpi.REQUIRED)

        self.addOutPort('compressed data', 'NPYarray')
        self.addOutPort('A', 'NPYarray')
        self.addOutPort('noise covariance', 'NPYarray')
        self.addOutPort('masked and normalized sense map')
    
    def validate(self):
        sense_map = self.getData('sensitivity map')
        if ( len(self.portEvents() ) > 0 ):
            self.setAttr('crop left', max=sense_map.shape[-1], min=1)
            self.setAttr('crop right', max=sense_map.shape[-1], min=1)
            self.setAttr('crop top', max=sense_map.shape[-2], min=1)
            self.setAttr('crop bottom', max=sense_map.shape[-2], min=1)
        if 'crop left' in self.widgetEvents():
            value_below = self.getVal('crop left')
            value_above = self.getVal('crop right')
            if value_below == sense_map.shape[-1]:
                self.setAttr('crop left', val=sense_map.shape[-1]-1)
            if value_above <= value_below:
                self.setAttr('crop right', val=value_below+1)
        if 'crop right' in self.widgetEvents():
            value_below = self.getVal('crop left')
            value_above = self.getVal('crop right')
            if value_above == 1:
                self.setAttr('crop right', val=2)
            if value_above <= value_below:
                self.setAttr('crop left', val=value_above-1)
        if 'crop top' in self.widgetEvents():
            value_below = self.getVal('crop top')
            value_above = self.getVal('crop bottom')
            if value_below == sense_map.shape[-2]:
                self.setAttr('crop top', val=sense_map.shape[-2]-1)
            if value_above <= value_below:
                self.setAttr('crop bottom', val=value_below+1)
        if 'crop bottom' in self.widgetEvents():
            value_below = self.getVal('crop top')
            value_above = self.getVal('crop bottom')
            if value_above == 1:
                self.setAttr('crop bottom', val=2)
            if value_above <= value_below:
                self.setAttr('crop top', val=value_above-1)

    def compute(self):
        import numpy as np
        from scipy import linalg

        self.log.node("Virtual Channels node running compute()")

        # GETTING WIDGET INFO
        image_ceiling = self.getVal('image ceiling')
        crop_left = self.getVal('crop left')
        crop_right = self.getVal('crop right')
        crop_top = self.getVal('crop top')
        crop_bottom = self.getVal('crop bottom')
        compute = self.getVal('compute')
        # number of virtual channels m
        m = self.getVal('virtual channels')

        # GETTING PORT INFO
        data = self.getData('data')
        noise = self.getData('noise')
        sensitivity_map_uncropped = self.getData('sensitivity map')
        
        

        # display sensitivity map to allow selection of ROI
        image = np.copy(sensitivity_map_uncropped)
        image = np.abs(image)
        image = np.square(image)
        image = np.mean(image, axis=0)
        image = np.sqrt(image)
        
        data_max = image.max()
        image = np.clip(image, 0.1, data_max)
        data_min = image.min()
        
        sensitivity_map_uncropped = np.divide( sensitivity_map_uncropped, image )
        mask = image > 0.02 * data_max
        sensitivity_map_uncropped = np.multiply( sensitivity_map_uncropped, mask )
        
        image[:, crop_left-1] = data_max
        image[:, crop_right-1] = data_max
        image[crop_top-1, :] = data_max
        image[crop_bottom-1, :] = data_max
        
        data_range = data_max - data_min
        new_max = data_range * 0.01 * image_ceiling + data_min
        dmask = np.ones(image.shape)
        image = np.minimum(image,new_max*dmask)
        if new_max > data_min:
            image = 255.*(image - data_min)/(new_max-data_min)
        red = green = blue = np.uint8(image)
        alpha = 255. * np.ones(blue.shape)
        h, w = red.shape[:2]
        image1 = np.zeros((h, w, 4), dtype=np.uint8)
        image1[:, :, 0] = red
        image1[:, :, 1] = green
        image1[:, :, 2] = blue
        image1[:, :, 3] = alpha

        format_ = QtGui.QImage.Format_RGB32
        
        image2 = QtGui.QImage(image1.data, w, h, format_)
        image2.ndarry = image1
        self.setAttr('image', val=image2)

        # crop sensitivity map
        sensitivity_map = sensitivity_map_uncropped[:,crop_top-1:crop_bottom,crop_left-1:crop_right]

        # get sizes
        # number of channels n
        n = sensitivity_map.shape[-3]
        x_size = sensitivity_map.shape[-1]
        y_size = sensitivity_map.shape[-2]
        nr_pixels = x_size * y_size

        if compute:

            # noise covariance matrix Psi
            noise_cv_matrix = np.cov(noise)

            # Cholesky decomposition to determine T, where T Psi T_H = 1
            L = np.linalg.cholesky(noise_cv_matrix)
            T = np.linalg.inv(L)

            # decorrelated sensitivity map S_hat
            S_hat = np.zeros([nr_pixels, n], dtype=np.complex64)
            for x in range(x_size):
                for y in range(y_size):
                    index = y + x * y_size
                    S_hat[index, :] = np.dot(T, sensitivity_map[:,y,x])
                        
            self.log.debug("after S_hat")
            
            # P = sum of S_hat S_hat_pseudo_inverse over all pixels
            P = np.zeros([n,n], dtype=np.complex64)
            S_hat_matrix = np.zeros([n,1], dtype=np.complex64)
            for index in range(nr_pixels):
                # pseudo inverse of S_hat
                S_hat_matrix[:,0] = S_hat[index,:]
                S_hat_pinv = np.linalg.pinv(S_hat_matrix)
                P = P + np.dot(S_hat_matrix, S_hat_pinv)
            self.log.debug("after S_hat_pinv")
            

            # singular value decomposition of P
            # if P is symmetric and positive definite, the SVD is P = U d U.H instead of P = U d V.H
            U, d, V = np.linalg.svd(P)
            self.log.debug("after svd")

            # the transformation matrix A is then given by A = C U.H T
            # C is diagonal matrix with 1 on the first m rows and 0 in the remaining
            # instead of using C, only assing mxn to A
            C = np.array(np.zeros([n,n]), dtype=np.float32)
            self.log.debug("after C")
            for x in range(m):
                C[x,x]=1.
            A_square = np.dot(C, np.dot(U.T.conjugate(), T))
            A = A_square[0:m,:]
            self.log.debug("after A")

            # Compress the data
            if data.ndim == 4:
                out = np.zeros([m,data.shape[-3],data.shape[-2],data.shape[-1]],dtype=data.dtype)
                for phase in range(data.shape[-3]):
                    for arm in range(data.shape[-2]):
                        for point in range(data.shape[-1]):
                            #print A.shape, out[:,phase,arm,point].shape, (np.dot(A, data[:,phase,arm,point])).shape
                            out[:,phase,arm,point] = np.dot(A, data[:,phase,arm,point])
            elif data.ndim == 3:
                out = np.zeros([m,data.shape[-2],data.shape[-1]],dtype=data.dtype)
                for arm in range(data.shape[-2]):
                    for point in range(data.shape[-1]):
                        out[:,arm,point] = np.dot(A, data[:,arm,point])


            # SETTING PORT INFO
            self.setData('compressed data', out)
            self.setData('A', A)
            self.setData('noise covariance', noise_cv_matrix)
    
            # end of compute
            self.setAttr('compute', val=False)
    
        self.setData('masked and normalized sense map', sensitivity_map_uncropped)

        return 0

    def execType(self):
        '''Could be GPI_THREAD, GPI_PROCESS, GPI_APPLOOP'''
        return gpi.GPI_APPLOOP #gpi.GPI_PROCESS  #Mike-debug: does it need to be an apploop for display widget?
