import numpy as np
from typing import List

# Constants
NUM_STATES = 87
MAX_ACTIONS = 5 

class MDP:
    def __init__(self, num_states = NUM_STATES, num_actions = MAX_ACTIONS):        
        self.num_states = num_states
        self.num_actions = num_actions
        self.reward_function = self.get_reward_function()
        self.transition_model = self.get_transition_model()

    def get_reward_function(self):
        reward_function = np.zeros(self.num_states)
        # Funcion especifica segun el problema
        reward_function[0] = 0.000000
        reward_function[1] = -0.787500
        reward_function[2] = -0.100000
        reward_function[3] = -150.000000
        reward_function[4] = -0.975000
        reward_function[5] = -0.893750
        reward_function[6] = -150.000000
        reward_function[7] = -0.893750
        reward_function[8] = -0.200000
        reward_function[9] = -0.800000
        reward_function[10] = 0.000000
        reward_function[11] = -1.062500
        reward_function[12] = -150.000000
        reward_function[13] = -1.062500
        reward_function[14] = -0.200000
        reward_function[15] = 0.000000
        reward_function[16] = -1.062500
        reward_function[17] = -0.200000
        reward_function[18] = 0.000000
        reward_function[19] = -0.200000
        reward_function[20] = -0.963636
        reward_function[21] = 0.000000
        reward_function[22] = -0.963636
        reward_function[23] = -500.000000
        reward_function[24] = -100.000000
        reward_function[25] = -100.000000
        reward_function[26] = 0.000000
        reward_function[27] = -100.000000
        reward_function[28] = -100.000000
        reward_function[29] = 0.000000
        reward_function[30] = -100.000000
        reward_function[31] = -100.000000
        reward_function[32] = 0.000000
        reward_function[33] = -100.000000
        reward_function[34] = -500.000000
        reward_function[35] = -100.000000
        reward_function[36] = -100.000000
        reward_function[37] = 0.000000
        reward_function[38] = -100.000000
        reward_function[39] = -100.000000
        reward_function[40] = 0.000000
        reward_function[41] = -100.000000
        reward_function[42] = -500.000000
        reward_function[43] = -100.000000
        reward_function[44] = -100.000000
        reward_function[45] = -100.000000
        reward_function[46] = -100.000000
        reward_function[47] = 0.000000
        reward_function[48] = 0.000000
        reward_function[49] = -100.000000
        reward_function[50] = -500.000000
        reward_function[51] = -100.000000
        reward_function[52] = -100.000000
        reward_function[53] = 0.000000
        reward_function[54] = -100.000000
        reward_function[55] = 0.000000
        reward_function[56] = -500.000000
        reward_function[57] = -100.000000
        reward_function[58] = 0.000000
        reward_function[59] = -500.000000
        reward_function[60] = -100.000000
        reward_function[61] = -100.000000
        reward_function[62] = 0.000000
        reward_function[63] = -500.000000
        reward_function[64] = -100.000000
        reward_function[65] = -100.000000
        reward_function[66] = 0.000000
        reward_function[67] = -500.000000
        reward_function[68] = -100.000000
        reward_function[69] = -100.000000
        reward_function[70] = 0.000000
        reward_function[71] = -500.000000
        reward_function[72] = -100.000000
        reward_function[73] = -100.000000
        reward_function[74] = -100.000000
        reward_function[75] = -100.000000
        reward_function[76] = -100.000000
        reward_function[77] = 0.000000
        reward_function[78] = -500.000000
        reward_function[79] = 0.000000
        reward_function[80] = -500.000000
        reward_function[81] = -100.000000
        reward_function[82] = -100.000000
        reward_function[83] = -100.000000
        reward_function[84] = 0.000000
        reward_function[85] = -500.000000
        reward_function[86] = -1000000.000000
        return reward_function
    
    def get_transition_model(self):
        transition_model = np.zeros((self.num_states, self.num_actions, self.num_states))
        # Funcion especifica segun el problema
        transition_model[0][0][1] = 1.00
        transition_model[0][2][2] = 1.00
        transition_model[0][4][3] = 1.00

        # transition_model[0][1][86] = 0.00
        # transition_model[0][3][86] = 0.00
        
        transition_model[1][0][4] = 1.00
        transition_model[1][2][5] = 1.00
        transition_model[1][4][6] = 1.00
        transition_model[2][0][7] = 1.00
        transition_model[2][2][8] = 1.00
        transition_model[2][3][9] = 1.00
        transition_model[3][0][3] = 1.00
        transition_model[3][1][3] = 1.00
        transition_model[3][2][3] = 1.00
        transition_model[3][3][3] = 1.00
        transition_model[3][4][3] = 1.00
        transition_model[4][0][10] = 1.00
        transition_model[4][2][11] = 1.00
        transition_model[4][4][12] = 1.00
        transition_model[5][0][13] = 1.00
        transition_model[5][2][14] = 1.00
        transition_model[5][3][15] = 1.00
        transition_model[7][0][16] = 1.00
        transition_model[7][2][17] = 1.00
        transition_model[7][3][18] = 1.00
        transition_model[8][0][19] = 1.00
        transition_model[8][3][20] = 1.00
        transition_model[9][0][21] = 1.00
        transition_model[9][2][22] = 1.00
        transition_model[9][4][23] = 1.00
        transition_model[11][0][24] = 1.00
        transition_model[11][2][25] = 1.00
        transition_model[11][3][26] = 1.00
        transition_model[13][0][27] = 1.00
        transition_model[13][2][28] = 1.00
        transition_model[13][3][29] = 1.00
        transition_model[14][0][30] = 1.00
        transition_model[14][3][31] = 1.00
        transition_model[15][0][32] = 1.00
        transition_model[15][2][33] = 1.00
        transition_model[15][4][34] = 1.00
        transition_model[16][0][35] = 1.00
        transition_model[16][2][36] = 1.00
        transition_model[16][3][37] = 1.00
        transition_model[17][0][38] = 1.00
        transition_model[17][3][39] = 1.00
        transition_model[18][0][40] = 1.00
        transition_model[18][2][41] = 1.00
        transition_model[18][4][42] = 1.00
        transition_model[19][0][43] = 1.00
        transition_model[19][3][44] = 1.00
        transition_model[20][0][45] = 1.00
        transition_model[20][2][46] = 1.00
        transition_model[20][3][47] = 1.00
        transition_model[21][0][48] = 1.00
        transition_model[21][2][49] = 1.00
        transition_model[21][4][50] = 1.00
        transition_model[22][0][51] = 1.00
        transition_model[22][2][52] = 1.00
        transition_model[22][3][53] = 1.00
        transition_model[25][0][54] = 1.00
        transition_model[26][0][55] = 1.00
        transition_model[26][4][56] = 1.00
        transition_model[28][0][57] = 1.00
        transition_model[29][0][58] = 1.00
        transition_model[29][4][59] = 1.00
        transition_model[30][0][60] = 1.00
        transition_model[31][0][61] = 1.00
        transition_model[32][0][62] = 1.00
        transition_model[32][4][63] = 1.00
        transition_model[33][0][64] = 1.00
        transition_model[36][0][65] = 1.00
        transition_model[37][0][66] = 1.00
        transition_model[37][4][67] = 1.00
        transition_model[38][0][68] = 1.00
        transition_model[39][0][69] = 1.00
        transition_model[40][0][70] = 1.00
        transition_model[40][4][71] = 1.00
        transition_model[41][0][72] = 1.00
        transition_model[43][0][73] = 1.00
        transition_model[44][0][74] = 1.00
        transition_model[45][0][75] = 1.00
        transition_model[46][0][76] = 1.00
        transition_model[47][0][77] = 1.00
        transition_model[47][4][78] = 1.00
        transition_model[48][0][79] = 1.00
        transition_model[48][4][80] = 1.00
        transition_model[49][0][81] = 1.00
        transition_model[51][0][82] = 1.00
        transition_model[52][0][83] = 1.00
        transition_model[53][0][84] = 1.00
        transition_model[53][4][85] = 1.00

        for a in range(self.num_states):
            for b in range(self.num_actions):
                if np.sum(transition_model[a][b]) == 0.0:
                    transition_model[a][b][86] = 1.0
        return transition_model


