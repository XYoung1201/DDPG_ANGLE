from ctypes import cdll, c_double, pointer,POINTER
import numpy as np
from drawState import plotAttack

class AttackEnv:
    def __init__(self):
        self.pSigma = 1/np.radians(90)
        self.pQ = 1/np.radians(90)
        self.pAction = 3
        self.lib = cdll.LoadLibrary('./Pattack_step.dll')
        self.lib.Step.argtypes = [c_double] + [POINTER(c_double)] * 3 + [c_double]
        self.figure = plotAttack()
        self.reset()
    
    def addPoints(self):
        self.t += self.step
        self.t_=np.append(self.t_,self.t)
        self.R_=np.append(self.R_,self.R)
        self.q_=np.append(self.q_,self.q)
        self.sigma_=np.append(self.sigma_,self.sigma)
        self.Action_ = np.append(self.Action_,self.Action)
        self.x_=np.append(self.x_,-self.R*np.cos(self.q))
        self.y_=np.append(self.y_,-self.R*np.sin(self.q))
    
    def reset(self):
        self.step = 3
        # self.sigma = np.radians(np.random.uniform(-20,20))
        self.sigma = 0
        self.R = np.random.uniform(30,50)
        self.q = np.radians(-np.random.uniform(0,30))
        self.v = 200
        self.t = 0
        self.Action = 0
        self.dead = False

        self.k1 = -1 #restrict_terminal_r
        self.k2 = -10 #restrict_terminal_q

        self.x_ = np.array([])
        self.y_ = np.array([])
        self.R_ = np.array([])
        self.sigma_ = np.array([]) #degree
        self.q_ = np.array([]) #degree
        self.Action_ = np.array([])
        self.t_ = np.array([])
        self.addPoints()
        return [self.pSigma*self.sigma,self.pQ*self.q]

    def Step(self,Action):

        self.Action = self.pAction*Action
        R_ptr = pointer(c_double(self.R))
        sigma_ptr = pointer(c_double(self.sigma))
        q_ptr = pointer(c_double(self.q))
        self.lib.Step(self.Action,R_ptr,sigma_ptr,q_ptr,self.step) 
        self.R = R_ptr.contents.value
        self.sigma = sigma_ptr.contents.value
        self.q = q_ptr.contents.value
        self.addPoints()

        done = False
        if self.R < self.step:
            done = True
        if abs(self.q)>np.pi/2 or abs(self.sigma)>np.pi/3 or self.t > 60:
            done = True
            self.dead = True

        return [self.pSigma*self.sigma,self.pQ*self.q],done
    
    def show(self,location):
        self.figure.plot_reset()
        self.figure.plot_subgraphs([self.x_,self.y_],[self.t_,self.Action_/self.R_],[self.t_,self.sigma_],[self.t_,self.q_],location)

    def getStates(self):

        # action_ = abs(self.Action_/self.R_)
        # action_max = np.max(action_)

        reward = self.k1*self.R+self.k2*abs(self.q) - (20 if self.dead else 0)
        reward/=10
        return self.pSigma*self.sigma_,self.pQ*self.q_,self.pAction*self.Action_,self.R,reward
    
    def close(self):
        pass