# from ctypes import cdll, c_double, pointer,POINTER
import PATTACK
import numpy as np
from drawState import plotAttack


class AttackEnv:
    def __init__(self):
        self.pSigma = 1 / np.radians(90)
        self.pQ = 1 / np.radians(90)
        self.pKr = 3
        self.pKq = 4
        # self.lib = cdll.LoadLibrary('./Pattack_step.dll')
        # self.lib.Step.argtypes = [c_double] + [POINTER(c_double)] * 3 + [c_double]
        self.figure = plotAttack()
        self.reset()

    def addPoints(self):
        self.t += self.step
        self.t_ = np.append(self.t_, self.t)
        self.R_ = np.append(self.R_, self.R)
        self.q_ = np.append(self.q_, self.q)
        self.sigma_ = np.append(self.sigma_, self.sigma)
        self.Action_ = np.append(self.Action_, self.Action)
        self.x_ = np.append(self.x_, -self.R * np.cos(self.q))
        self.y_ = np.append(self.y_, -self.R * np.sin(self.q))
        self.kr_ = np.append(self.kr_, -self.R * np.sin(self.kr))
        self.kq_ = np.append(self.kq_, -self.R * np.sin(self.kq))

    def reset(self):
        self.step = 1
        # self.sigma = np.radians(np.random.uniform(-20,20))
        self.sigma = 0
        self.R = np.random.uniform(30, 50)
        self.q = np.radians(-np.random.uniform(0, 30))
        self.v = 200
        self.t = 0
        self.Action = 0
        self.dead = False
        self.kq = 0
        self.kr = 0

        self.k1 = -1  # restrict_terminal_r
        self.k2 = -50  # restrict_terminal_q
        self.k3 = -10  # restrct_terminal_sigma
        self.k4 = -5  # restrct_process_a

        self.x_ = np.array([])
        self.y_ = np.array([])
        self.kr_ = np.array([])
        self.kq_ = np.array([])
        self.R_ = np.array([])
        self.sigma_ = np.array([])  # degree
        self.q_ = np.array([])  # degree
        self.Action_ = np.array([])
        self.t_ = np.array([])
        self.addPoints()
        return [self.pSigma * self.sigma, self.pQ * self.q]

    def Step(self, Action):
        self.kr = self.pKr * (Action[0] + 1)
        self.kq = self.pKq * (Action[1] + 1)
        self.Action = -self.kr * np.sin(self.sigma) + self.kq * self.q
        # R_ptr = pointer(c_double(self.R))
        # sigma_ptr = pointer(c_double(self.sigma))
        # q_ptr = pointer(c_double(self.q))
        self.R, self.sigma, self.q = PATTACK.Step(
            self.Action, self.R, self.sigma, self.q, self.step
        )
        # self.R = R_ptr.contents.value
        # self.sigma = sigma_ptr.contents.value
        # self.q = q_ptr.contents.value
        self.addPoints()

        done = False
        if (
            abs(self.R) < self.step
            and abs(self.q) < np.radians(self.step)
            and abs(self.sigma) < np.radians(self.step)
        ):
            done = True
        if (
            self.R < self.step
            or abs(self.q) > np.deg2rad(90)
            or abs(self.sigma) > np.deg2rad(60)
            or self.t > 70
        ):
            done = True
            self.dead = True
        return [self.pSigma * self.sigma, self.pQ * self.q], done

    def show(self, location):
        self.figure.plot_reset()
        self.figure.plot_subgraphs(
            [self.x_, self.y_],
            [self.t_, self.Action_ / self.R_],
            [self.t_, self.sigma_],
            [self.t_, self.q_],
            location,
        )
        self.figure.plot_reset()
        self.figure.plot_subgraphs(
            [self.x_, self.y_],
            [self.t_, self.kr_],
            [self.t_, self.kq_],
            [self.t_, self.q_],
            location,
        )

    def getStates(self):
        action_ = abs(self.Action_ / self.R_)
        if not self.dead:
            self.R = 0
            self.q = 0
            self.sigma = 0

        reward = (
            +(30 if self.R < self.step else 0)
            +(60 if self.q < np.radians(self.step/2) else 0)
            +self.k1 * abs(self.R)
            + self.k2 * abs(self.q)
            + self.k3 * abs(self.sigma)
            # + self.k4 * abs(action_.max() + action_.mean())
            - (50 if self.dead else 0)
        )
        reward /= 10
        return (
            self.pSigma * self.sigma_,
            self.pQ * self.q_,
            self.kr_ / self.pKr - 1,
            self.kq_ / self.pKq - 1,
            self.R,
            reward,
        )

    def close(self):
        pass

    def resetTest(self, sigma, R, q, sim_step):
        self.step = sim_step
        # self.sigma = np.radians(np.random.uniform(-20,20))
        self.sigma = sigma
        self.R = R
        self.q = q
        self.v = 200
        self.t = 0
        self.Action = 0
        self.dead = False
        self.kq = 0
        self.kr = 0

        self.k1 = -1  # restrict_terminal_r
        self.k2 = -50  # restrict_terminal_q
        self.k3 = -10  # restrct_terminal_sigma
        self.k4 = -5  # restrct_process_a

        self.x_ = np.array([])
        self.y_ = np.array([])
        self.kr_ = np.array([])
        self.kq_ = np.array([])
        self.R_ = np.array([])
        self.sigma_ = np.array([])  # degree
        self.q_ = np.array([])  # degree
        self.Action_ = np.array([])
        self.t_ = np.array([])
        self.addPoints()
        return [self.pSigma * self.sigma, self.pQ * self.q]

    def StepTest(self, Action):
        self.kr = self.pKr * (Action[0] + 1)
        self.kq = self.pKq * (Action[1] + 1)
        self.Action = -self.kr * np.sin(self.sigma) + self.kq * self.q
        # R_ptr = pointer(c_double(self.R))
        # sigma_ptr = pointer(c_double(self.sigma))
        # q_ptr = pointer(c_double(self.q))
        self.R, self.sigma, self.q = PATTACK.Step(
            self.Action, self.R, self.sigma, self.q, self.step
        )
        # self.R = R_ptr.contents.value
        # self.sigma = sigma_ptr.contents.value
        # self.q = q_ptr.contents.value
        self.addPoints()

        done = False
        if (
            abs(self.R) < self.step
            and abs(self.q) < np.radians(self.step)
            and abs(self.sigma) < np.radians(self.step)
        ):
            done = True
        if (
            abs(self.q) > np.deg2rad(90)
            or abs(self.sigma) > np.deg2rad(60)
            or self.t > 70
        ):
            done = True
            self.dead = True
        return (
            [self.pSigma * self.sigma, self.pQ * self.q],
            done,
            self.R,
            np.rad2deg(self.sigma),
            np.rad2deg(self.q),
            self.Action / self.R * self.v,
            self.t,
        )
