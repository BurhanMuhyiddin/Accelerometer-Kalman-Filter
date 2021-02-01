from http.server import HTTPServer, BaseHTTPRequestHandler

import sys
import threading
import numpy as np
import matplotlib.pyplot as plt

all_acc_data = np.zeros((1,3))
crt_acc_data = np.zeros((1,3))
prd_acc_data = np.zeros((1,3))
flag = False


class KalmanFilter():
    
    def __init__(self):
        self.d_t = 1/20 # milliseconds
        self.A = np.array([ 
                            [1,   0,   0,   self.d_t,           0,                 0, 0.5*self.d_t*self.d_t,   0,                                          0],
                            [0,   1,   0,          0,    self.d_t,                 0, 0,                       0.5*self.d_t*self.d_t,                      0],
                            [0,   0,   1,          0,           0,          self.d_t, 0,                       0,                      0.5*self.d_t*self.d_t],
                            [0,   0,   0,          1,           0,                 0, self.d_t,                0,                                          0],
                            [0,   0,   0,          0,           1,                 0, 0,                       self.d_t,                                   0],
                            [0,   0,   0,          0,           0,                 1, 0,                       0,                                   self.d_t],
                            [0,   0,   0,          0,           0,                 0, 1,                       0,                                          0],
                            [0,   0,   0,          0,           0,                 0, 0,                       1,                                          0],
                            [0,   0,   0,          0,           0,                 0, 0,                       0,                                          1]
                          ])

        self.C = np.array([
                            [0,   0,   0,   0,   0,   0,   1,   0,   0],
                            [0,   0,   0,   0,   0,   0,   0,   1,   0],
                            [0,   0,   0,   0,   0,   0,   0,   0,   1]
                          ])

        self.P_t = np.identity(9)*0.2
        self.Q_t = np.identity(9)
        self.R_t = np.identity(3)*5
        self.x_hat_t = np.zeros((9,1))

    def run(self):
        akf_th = threading.Thread(target=self.apply_kalman_filter)
        akf_th.start()
        akf_th.join()

    def apply_kalman_filter(self):
        global flag, crt_acc_data, prd_acc_data
        #while not flag:
        self.predict_stage()
        self.update_stage(crt_acc_data.transpose())
        #print(np.shape(self.x_hat_t))
        prd_acc_data = np.append(prd_acc_data, self.x_hat_t[-3:,:].transpose(), axis=0)
        prd_acc_data = prd_acc_data[-50:,:]

    def predict_stage(self):
        self.x_hat_t = self.A.dot(self.x_hat_t)
        self.P_t = self.A.dot(self.P_t).dot(self.A.transpose()) + self.Q_t

    def update_stage(self, y_t):
        K_t = self.P_t.dot(self.C.transpose()).dot(np.linalg.inv(self.C.dot(self.P_t).dot(self.C.transpose())+self.R_t))
        self.x_hat_t = self.x_hat_t + K_t.dot(y_t - self.C.dot(self.x_hat_t))
        self.P_t = self.P_t - K_t.dot(self.C).dot(self.P_t)

kf = KalmanFilter()

class Serv(BaseHTTPRequestHandler):

    def do_GET(self):
        global all_acc_data, crt_acc_data, kf
        
        self.send_response(200)
        self.send_header('content_type', 'text/html')
        self.end_headers()
        _request = self.path[1:].split(",")
        ind=0
        for req in _request:
            crt_acc_data[0,ind] = float(req)
            ind+=1
        all_acc_data = np.append(all_acc_data, crt_acc_data, axis=0)
        all_acc_data = all_acc_data[-50:,:]
        kf.apply_kalman_filter()

def init_web_server():
    httpd = HTTPServer(('192.168.1.75', 8080), Serv)
    httpd.serve_forever()

def press(event):
    global flag
    sys.stdout.flush()
    if event.key == 'q':
        flag = True

def plot_signal():
    global all_acc_data, flag, prd_acc_data

    # plot related initializations
    plt.close('all')
    fig, _ = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', press)
    plt.ion()
    plt.show()
    ###############################
    
    while not flag:   
        plt.cla()
        plt.xlim(left=5, right=50)

        print(prd_acc_data[:,0])
        
        plt.plot(all_acc_data[:,0])
        plt.plot(prd_acc_data[:,0])

        #plt.plot(all_acc_data[:,1])
        #plt.plot(prd_acc_data[:,1])

        #plt.plot(all_acc_data[:,2])
        #plt.plot(prd_acc_data[:,2])

        #plt.legend(["o_x","p_x","o_y","p_y","o_z","p_z"])

        plt.pause(0.01)

def main():
    
    data_get_th = threading.Thread(target=init_web_server)
    data_get_th.start()

    plot_signal_th = threading.Thread(target=plot_signal)
    plot_signal_th.start()

    data_get_th.join()
    plot_signal_th.join()

if __name__ == '__main__':
    main()


# A matrix
# 1   0   0   t   0   0   
# 0   1   0   0   t   0   
# 0   0   1   0   0   t

# B matrix
# 0.5t^2   0        0
# 0        0.5t^2   0
# 0        0        0.5t^2

# C matrix
# 0   0   0   0   0   0   1   0   0
# 0   0   0   0   0   0   0   1   0  
# 0   0   0   0   0   0   0   0   1