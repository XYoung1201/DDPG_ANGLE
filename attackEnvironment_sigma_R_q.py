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
        self.x_=np.append(self.x_,-self.R*np.cos(self.q))
        self.y_=np.append(self.y_,-self.R*np.sin(self.q))
    
    def reset(self):
        self.step = 1
        self.sigma = 0.0
        self.R = np.random.uniform(40,50)
        self.q = np.random.uniform(-30,30)/57.3
        self.t = 0
        self.Action = 0

        self.k1 = -1 #restrict_terminal_r
        self.k2 = -10 #restrict_terminal_q
        self.k3 = -10 #restrict_terminal_sigma
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
        self.addPoints()
        return [self.sigma,self.q]

    def Step(self,Action):
        # if abs(self.sigma)>self.sigma_max:
        #     self.n2_ = self.n2_ + 1
        # if abs(Action*self.v/self.R)>self.a_max:
        #     self.n1_ = self.n1_ + 1

        self.Action = Action
        R_ptr = pointer(c_double(self.R))
        sigma_ptr = pointer(c_double(self.sigma))
        q_ptr = pointer(c_double(self.q))
        self.lib.Step(self.Action,R_ptr,sigma_ptr,q_ptr,self.step) 
        self.R = R_ptr.contents.value
        self.sigma = sigma_ptr.contents.value
        self.q = q_ptr.contents.value
        self.addPoints()

        done = False
        if self.R < self.step or abs(self.q)>np.pi/2 or abs(self.sigma)>np.pi*1/3 or self.t > 80:
            done = True

        return [self.sigma,self.q],done
    
    def show(self,location):
        self.figure.plot_reset()
        self.figure.plot_subgraphs([self.x_,self.y_],[self.t_,self.Action_],[self.t_,self.sigma_],[self.t_,self.q_],location)

    def getStates(self):
        if self.R<self.step:
            self.R = 0
        else:
            self.R -= self.step
        if abs(self.sigma) < 1/57.3:
            self.sigma = 0
        else:
            self.sigma = abs(self.sigma) - 1/57.3
        if abs(self.q) < 1/57.3:
            self.q = 0
        else:
            self.q = abs(self.q) - 1/57.3
        reward = self.k1*abs(self.R)+self.k2*abs(self.q) + self.k3*abs(self.sigma)
        return [self.sigma_,self.q_],self.Action_,reward,self.R,self.q*57.3,self.sigma*57.3
    
    def close(self):
        pass