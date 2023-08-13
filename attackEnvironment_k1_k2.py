from ctypes import cdll, c_double, pointer,POINTER
import numpy as np
from drawState import plotAttack

class AttackEnv:
    def __init__(self):
        self.lib = cdll.LoadLibrary('./Pattack_step.dll')
        self.lib.Step.argtypes = [c_double] + [POINTER(c_double)] * 3 + [c_double]
        self.figure = plotAttack()
        self.reset()
    
    def addPoints(self):
        self.t += self.step
        self.t_=np.append(self.t_,self.t)
        self.R_=np.append(self.R_,self.R)
        self.q_=np.append(self.q_,self.q*57.3)
        self.sigma_=np.append(self.sigma_,self.sigma*57.3)
        self.Action_ = np.append(self.Action_,self.Action)
        self.kr_ = np.append(self.kr_,self.kr)
        self.kq_ = np.append(self.kq_,self.kq)
        self.x_=np.append(self.x_,-self.R*np.cos(self.q))
        self.y_=np.append(self.y_,-self.R*np.sin(self.q))
    
    def reset(self):
        self.step = 1
        self.sigma = np.random.uniform(-np.pi*1/6,np.pi*1/6)
        self.R = np.random.uniform(30,50)
        self.q = -np.random.uniform(0,30)/57.3
        self.v = 200
        self.t = 0
        self.Action = 0
        self.kr = 3
        self.kq = 0.1

        self.k1 = -1 #restrict_terminal_r
        self.k2 = -10 #restrict_terminal_q
        self.k3 = -10 #restrict_terminal_sigma
        self.k4 = -1
        self.k5 = -1
        # self.kn_Action_max = 3 #Action_max
        # self.kn_sigma_max = 1 #sigma_max

        # self.n_Action_max = np.array([0])
        # self.n_sigma_max = np.array([0])
        self.x_ = np.array([])
        self.y_ = np.array([])
        self.R_ = np.array([])
        self.sigma_ = np.array([]) #degree
        self.q_ = np.array([]) #degree
        self.Action_ = np.array([])
        self.t_ = np.array([])
        self.kr_ = np.array([])
        self.kq_ = np.array([])
        self.addPoints()
        return [self.sigma/(np.pi/3),self.q/(np.pi/2)]

    def Step(self,Action):
        # if abs(self.sigma)>self.sigma_max:
        #     self.n2_ = self.n2_ + 1
        # if abs(Action*self.v/self.R)>self.a_max:
        #     self.n1_ = self.n1_ + 1

        self.kr = Action[0]*4+5
        self.kr = 3
        self.kq = Action[1]*2+2
        # self.kq = 0

        # if(self.R<20):
        #     self.step = 0.5
        # if(self.R<10):
        #     self.step = 0.1
        # elif self.R>5:
        #     self.step = 0.1
        # elif self.R>1:
        #     self.step = 0.1
        # elif self.R>0.1:
        #     self.step = 0.01

        self.Action = -self.kr*np.sin(self.sigma)+self.kq*self.q
        R_ptr = pointer(c_double(self.R))
        sigma_ptr = pointer(c_double(self.sigma))
        q_ptr = pointer(c_double(self.q))
        self.lib.Step(self.Action,R_ptr,sigma_ptr,q_ptr,self.step) 
        self.R = R_ptr.contents.value
        self.sigma = sigma_ptr.contents.value
        self.q = q_ptr.contents.value
        self.addPoints()

        done = False
        if self.R < self.step or abs(self.q)>np.pi/2 or abs(self.sigma)>np.pi/3 or self.t > 80:
            done = True

        return [self.sigma/(np.pi/3),self.q/(np.pi/2)],done
    
    def show(self,location):
        self.figure.plot_reset()
        self.figure.plot_subgraphs([self.x_,self.y_],[self.t_,self.Action_/self.R_],[self.t_,self.sigma_],[self.t_,self.q_],location)

    def getStates(self):
        if self.R<self.step:
            self.R = -4
        else:
            self.R -= self.step
            # self.R = -100
        if abs(self.sigma) < self.step*2/57.3:
            self.sigma = -4
        else:
            self.sigma = abs(self.sigma) - 5/57.3
            # self.sigma = -100
        if abs(self.q) < 5/57.3:
            self.q = -4
        else:
            self.q = abs(self.q) - self.step*2/57.3
            # self.q = -100

        q_mean = np.mean(abs(self.q_))

        action_ = abs(self.Action_/self.R_)
        action_max = np.max(action_)
        # action_mean = np.mean(action_)
        reward = self.k1*self.R+self.k2*self.q + self.k2*q_mean + self.k3*self.sigma # + self.k4*action_max  # + self.k5*action_mean
        # reward = self.k1*self.R+self.k2*self.q + self.k4*action_max + self.k5*action_mean
        reward /= 10
        return self.sigma_/(np.pi/3),self.q_/(np.pi/2),self.Action_,(self.kr_-5)/4,(self.kq_-2)/2,reward,self.R,self.q*57.3,self.sigma*57.3
    
    def close(self):
        pass