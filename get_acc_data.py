from http.server import HTTPServer, BaseHTTPRequestHandler

import sys
import threading
import numpy as np
import matplotlib.pyplot as plt

all_acc_data = np.zeros((1,3))

class Serv(BaseHTTPRequestHandler):

    def do_GET(self):
        global all_acc_data
        crt_acc_data = np.zeros((1,3))
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

def init_web_server():
    httpd = HTTPServer(('192.168.1.75', 8080), Serv)
    httpd.serve_forever()

class KalmanFilter():
    def __init__(self):
        global all_acc_data
        self.flag = False

        # plot related initializations
        plt.close('all')
        self.fig, _ = plt.subplots()
        self.fig.canvas.mpl_connect('key_press_event', self.press)
        plt.ion()
        plt.show()

    def press(self, event):
        sys.stdout.flush()
        if event.key == 'q':
            self.flag = True

    def plot_signal(self):
        global all_acc_data
        
        while not self.flag:   
            plt.cla()
            plt.xlim(left=5, right=50)
            plt.plot(all_acc_data[:,0])
            plt.plot(all_acc_data[:,1])
            plt.plot(all_acc_data[:,2])
            plt.pause(0.01)

def main():
    kf = KalmanFilter()
    data_get = threading.Thread(target=init_web_server)
    data_get.start()
    kf.plot_signal()
    data_get.join()

if __name__ == '__main__':
    main()