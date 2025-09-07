import numpy as np
import math
from numba import njit
import traceback

DEBUG_PRINT=True

@njit
def F(r, Q):
    return (Q**2/r**3-1/r)

@njit
def DF(r, Q):
    return (-3/2*Q**2/r**5+1/2/r**3)
    
@njit
def Quv_func(x):
    if x > 500:
        return x
    return np.log(1+np.exp(x))
    
@njit
def cut_trail(f_str):
    cut = 0
    for c in f_str[::-1]:
        if c == "0":
            cut += 1
        else:
            break
    if cut == 0:
        for c in f_str[::-1]:
            if c == "9":
                cut += 1
            else:
                cut -= 1
                break
    if cut > 0:
        f_str = f_str[:-cut]
    if f_str == "":
        f_str = "0"
    return f_str


@njit
def float2str(value):
    if math.isnan(value):
        return "nan"
    elif value == 0.0:
        return "0.0"
    elif value < 0.0:
        return "-" + float2str(-value)
    elif math.isinf(value):
        return "inf"
    else:
        max_digits = 16
        min_digits = -4
        e10 = math.floor(math.log10(value)) if value != 0.0 else 0
        if min_digits < e10 < max_digits:
            i_part = math.floor(value)
            f_part = math.floor((1 + value % 1) * 10.0 ** max_digits)
            i_str = str(i_part)
            f_str = cut_trail(str(f_part)[1:max_digits - e10])
            return i_str + "." + f_str
        else:
            m10 = value / 10.0 ** e10
            exp_str_len = 4
            i_part = math.floor(m10)
            f_part = math.floor((1 + m10 % 1) * 10.0 ** max_digits)
            i_str = str(i_part)
            f_str = cut_trail(str(f_part)[1:max_digits])
            e_str = str(e10)
            if e10 >= 0:
                e_str = "+" + e_str
            return i_str + "." + f_str + "e" + e_str
            
@njit
def pde_solve_point(R, S, Q, dudv, i, j, source_r=0, source_s=0):
    midpoint_R = (R[i-1,j]+R[i,j-1])/2
    midpoint_S = (S[i-1,j]+S[i,j-1])/2
    midpoint_r = np.sqrt(midpoint_R)
    exps = np.exp(midpoint_S)
    Ruv = (F(midpoint_r, Q)+source_r) * exps
    Suv = (DF(midpoint_r, Q)+source_s) * exps
    R[i,j] = R[i-1,j] + R[i,j-1] - R[i-1,j-1] + dudv * Ruv
    S[i,j] = S[i-1,j] + S[i,j-1] - S[i-1,j-1] + dudv * Suv
    
@njit
def pde_solve_point_sourcesR(R, S, Q, dudv, i, j, source_r, source_s,source_rr,source_rs):
    midpoint_R = (R[i-1,j]+R[i,j-1])/2
    midpoint_S = (S[i-1,j]+S[i,j-1])/2
    midpoint_r = np.sqrt(midpoint_R)
    exps = np.exp(midpoint_S)
    Ruv = (F(midpoint_r, Q)+source_r+source_rr*midpoint_r) * exps
    Suv = (DF(midpoint_r, Q)+source_s+source_rs*midpoint_r) * exps
    R[i,j] = R[i-1,j] + R[i,j-1] - R[i-1,j-1] + dudv * Ruv
    S[i,j] = S[i-1,j] + S[i,j-1] - S[i-1,j-1] + dudv * Suv
    
@njit
def pde_solve_point_Qr(R, S, Q, dudv, i, j, source_r=0, source_s=0,Qr=0):
    midpoint_R = (R[i-1,j]+R[i,j-1])/2
    midpoint_S = (S[i-1,j]+S[i,j-1])/2
    midpoint_r = np.sqrt(midpoint_R)
    Q += Qr * midpoint_r
    exps = np.exp(midpoint_S)
    Ruv = (F(midpoint_r, Q)+source_r) * exps
    Suv = (DF(midpoint_r, Q)+source_s) * exps
    R[i,j] = R[i-1,j] + R[i,j-1] - R[i-1,j-1] + dudv * Ruv
    S[i,j] = S[i-1,j] + S[i,j-1] - S[i-1,j-1] + dudv * Suv
    
@njit
def pde_solve_point_skip(Rcurr,Rprev, Scurr, Sprev, Q, dudv, i, j, source_r, source_s,source_rr,source_rs):
    midpoint_R = (Rprev[j]+Rcurr[j-1])/2
    midpoint_S = (Sprev[j]+Scurr[j-1])/2
    midpoint_r = np.sqrt(midpoint_R)
    exps = np.exp(midpoint_S)
    Ruv = (F(midpoint_r, Q)+source_r+source_rr*midpoint_r) * exps
    Suv = (DF(midpoint_r, Q)+source_s+source_rs*midpoint_r) * exps
    Rcurr[j] = Rprev[j] + Rcurr[j-1] - Rprev[j-1] + dudv * Ruv
    Scurr[j] = Sprev[j] + Scurr[j-1] - Sprev[j-1] + dudv * Suv
           
@njit
def pde_solve_point_sources5(Rcurr,Rprev, Scurr, Sprev, Q, dudv, i, j, source_r, source_s,source_rr,source_rs,sources_r5,sources_s5):
    midpoint_R = (Rprev[j]+Rcurr[j-1])/2
    midpoint_S = (Sprev[j]+Scurr[j-1])/2
    midpoint_r = np.sqrt(midpoint_R)
    exps = np.exp(midpoint_S)
    Ruv = (F(midpoint_r, Q)+source_r+source_rr*midpoint_r) * exps+sources_r5
    Suv = (DF(midpoint_r, Q)+source_s+source_rs*midpoint_r) * exps+sources_s5
    Rcurr[j] = Rprev[j] + Rcurr[j-1] - Rprev[j-1] + dudv * Ruv
    Scurr[j] = Sprev[j] + Scurr[j-1] - Sprev[j-1] + dudv * Suv
    
@njit
def pde_solve_internal(R, S, Q, du, dv):
    dudv = du * dv
    for i in range(1,R.shape[0]):
        for j in range(1,R.shape[1]):
            pde_solve_point(R,S,Q,dudv,i,j)
            
    return R,S
    
@njit
def pde_solve_diagonal_internal(R, S, Q, du, dv):
    dudv = du * dv
    length = R.shape[0]
    for i in range(1,length):
        for j in range(length-i,length):
            pde_solve_point(R,S,Q,dudv,i,j)
            
    return R,S

@njit
def pde_solve_diagonal2_internal(R, S, Q, du, dv):
    dudv = du * dv
    length = R.shape[0]
    for i in range(1,length):
        for j in range(length-i+1,length):
            pde_solve_point(R,S,Q,dudv,i,j)

    return R,S

@njit
def pde_solve_with_fluxes_internal(R, S, Q, du, dv,source_r,source_s):
    dudv = du * dv
    length = R.shape[0]
    for i in range(1,length):
        for j in range(length-i+1,length):
            pde_solve_point(R,S,Q,dudv,i,j,source_r,source_s)

    return R,S
    
@njit
def pde_solve_with_sources_internal(R, S, Q, du, dv,sources_r,sources_s):
    dudv = du * dv
    length = R.shape[0]
    for i in range(1,length):
        for j in range(length-i+1,length):
            pde_solve_point(R,S,Q,dudv,i,j,(sources_r[i]+sources_r[i-1])/2,(sources_s[i]+sources_s[i-1])/2)

    return R,S    

@njit
def pde_solve_with_charge_internal(R, S, Qv, du, dv,sources_r,sources_s):
    dudv = du * dv
    length = R.shape[0]
    for i in range(1,length):
        for j in range(length-i+1,length):
            pde_solve_point(R,S,(Qv[i]+Qv[i-1])/2,dudv,i,j,(sources_r[i]+sources_r[i-1])/2,(sources_s[i]+sources_s[i-1])/2)

    return R,S    

@njit
def pde_solve_with_charge_Quv_internal(R, S, Qv, du, dv,sources_r,sources_s, Quv,delta_u_v):
    dudv = du * dv
    length = R.shape[0]
    for i in range(1,length):
        for j in range(length-i+1,length):
            Q = (Qv[i]+Qv[i-1])/2 + (Quv[i]+Quv[i-1])/2*Quv_func(dv * (i-1/2) + du * (j-1/2)+delta_u_v)
            pde_solve_point(R,S,Q,dudv,i,j,(sources_r[i]+sources_r[i-1])/2,(sources_s[i]+sources_s[i-1])/2)

    return R,S  
    
@njit
def pde_solve_with_sources_r_internal(R, S, Qv, du, dv,sources_r,sources_s, Quv,delta_u_v,sources_rr,sources_rs):
    dudv = du * dv
    length = R.shape[0]
    for i in range(1,length):
        for j in range(length-i+1,length):
            Q = (Qv[i]+Qv[i-1])/2 + (Quv[i]+Quv[i-1])/2*Quv_func(dv * (i-1/2) + du * (j-1/2)+delta_u_v)
            pde_solve_point_sourcesR(R,S,Q,dudv,i,j,(sources_r[i]+sources_r[i-1])/2,(sources_s[i]+sources_s[i-1])/2,
            (sources_rr[i]+sources_rr[i-1])/2,(sources_rs[i]+sources_rs[i-1])/2)

    return R,S
    
   
@njit
def pde_solve_with_charge_Qr_internal(R, S, Qv, du, dv,sources_r,sources_s, Qr,Qr0,delta_u_v):
    dudv = du * dv
    length = R.shape[0]
    for i in range(1,length):
        for j in range(length-i+1,length):
            Q = (Qv[i]+Qv[i-1])/2 + (Qr0[i]+Qr0[i-1])/2 *(dv * (i-1/2) + du * (j-1/2)+delta_u_v)
            pde_solve_point_Qr(R,S,Q,dudv,i,j,(sources_r[i]+sources_r[i-1])/2,(sources_s[i]+sources_s[i-1])/2,(Qr[i]+Qr[i-1])/2*(dv * (i-1/2) + du * (j-1/2) + delta_u_v))

    return R,S
    
@njit
def pde_solve_skip_internal(R, S, length, skip, R_init, R2_init, S_init, S2_init, Qv, du, dv, sources_r, sources_s, Quv, delta_u_v, sources_rr, sources_rs):
    dudv = du * dv
    l=length
    skipped_l = (l-1)//skip+1
    # R derivatives
    Rv = np.zeros((skipped_l,skipped_l),dtype=np.float64)
    Rvv = np.zeros((skipped_l,skipped_l),dtype=np.float64)
    Ru = np.zeros((skipped_l,skipped_l),dtype=np.float64)
    Ruu = np.zeros((skipped_l,skipped_l),dtype=np.float64)
    # S derivatives
    Sv = np.zeros((skipped_l,skipped_l),dtype=np.float64)
    Svv = np.zeros((skipped_l,skipped_l),dtype=np.float64)
    Su = np.zeros((skipped_l,skipped_l),dtype=np.float64)
    Suu = np.zeros((skipped_l,skipped_l),dtype=np.float64)
    
    Rcurr=np.zeros(l,dtype=np.float64)
    Rprev=np.zeros(l,dtype=np.float64)
    Rprev2=np.zeros(l,dtype=np.float64)
    Scurr=np.zeros(l,dtype=np.float64)
    Sprev=np.zeros(l,dtype=np.float64)
    Sprev2=np.zeros(l,dtype=np.float64)
    Rprev[l-1]=R2_init[0]
    Sprev[l-1]=S2_init[0]
    if (l-1)%skip == 0:
        R[0,(l-1)//skip]=R2_init[0]
        S[0,(l-1)//skip]=S2_init[0]
    #R[i,l-i-1] = np.float64(R2_init[i])
    #S[i,l-i-1] = np.float64(S2_init[i])
    #if i + 1 < l:
    #    R[i+1,l-i-1] = np.float64(R_init[i])
    #    S[i+1,l-i-1] = np.float64(S_init[i])
    #equivalent to:
    #    R[i,l-i] = np.float64(R_init[i-1])
    #    S[i,l-i] = np.float64(S_init[i-1])
    
    for i in range(1,l):
        #R[i,l-i-1] = np.float64(R2_init[i])
        #S[i,l-i-1] = np.float64(S2_init[i])
        Rcurr[l-i-1]=R2_init[i]
        Scurr[l-i-1]=S2_init[i]
        #R[i,l-i] = np.float64(R_init[i-1])
        #S[i,l-i] = np.float64(S_init[i-1])
        Rcurr[l-i] = R_init[i-1]
        Scurr[l-i] = S_init[i-1]
        for j in range(l-i+1,l):
            Q = (Qv[i]+Qv[i-1])/2 + (Quv[i]+Quv[i-1])/2*Quv_func(dv * (i-1/2) + du * (j-1/2)+delta_u_v)
            pde_solve_point_skip(Rcurr,Rprev, Scurr, Sprev, Q,dudv,i,j,(sources_r[i]+sources_r[i-1])/2,(sources_s[i]+sources_s[i-1])/2,
            (sources_rr[i]+sources_rr[i-1])/2,(sources_rs[i]+sources_rs[i-1])/2)
                
        if (i % skip) == 0:
            row = i // skip
            j = ((l-i-1)//skip) * skip
            for col in range((l-i-1)//skip,skipped_l):
                R[row,col]=Rcurr[j]
                S[row,col]=Scurr[j]
                if j > 0 and j+1<l:
                   Ru[row,col] = (Rcurr[j+1]-Rcurr[j-1]) / (2 * du)
                   Ruu[row,col] = (Rcurr[j+1]-2*Rcurr[j]+Rcurr[j-1]) / (du*du)
                   Su[row,col] = (Scurr[j+1]-Scurr[j-1]) / (2 * du)
                   Suu[row,col] = (Scurr[j+1]-2*Scurr[j]+Scurr[j-1]) / (du*du)
                j += skip
        if (i-1)%skip == 0:
            row = (i-1) // skip
            j = ((l-i-1)//skip) * skip
            for col in range((l-i-1)//skip,skipped_l):
               Rv[row,col] = (Rcurr[j]-Rprev2[j]) / (2 * dv)
               Rvv[row,col] = (Rcurr[j]-2*Rprev[j]+Rprev2[j]) / (dv*dv)
               Sv[row,col] = (Scurr[j]-Sprev2[j]) / (2 * dv)
               Svv[row,col] = (Scurr[j]-2*Sprev[j]+Sprev2[j]) / (dv*dv)
               j += skip
        Rcurr,Rprev,Rprev2 = Rprev2,Rcurr,Rprev
        Scurr,Sprev,Sprev2 = Sprev2,Scurr,Sprev
    return R,S,Rv,Rvv,Ru,Ruu,Sv,Svv,Su,Suu
                
@njit
def pde_solve_sources5_internal(R, S, length, skip, R_init, R2_init, S_init, S2_init, Qv, du, dv, sources_r, sources_s, Quv, delta_u_v, sources_rr, sources_rs,sources_r5,sources_s5,delta_u_v_5):
    dudv = du * dv
    l=length
    skipped_l = (l-1)//skip+1
    # R derivatives
    Rv = np.zeros((skipped_l,skipped_l),dtype=np.float64)
    Rvv = np.zeros((skipped_l,skipped_l),dtype=np.float64)
    Ru = np.zeros((skipped_l,skipped_l),dtype=np.float64)
    Ruu = np.zeros((skipped_l,skipped_l),dtype=np.float64)
    # S derivatives
    Sv = np.zeros((skipped_l,skipped_l),dtype=np.float64)
    Svv = np.zeros((skipped_l,skipped_l),dtype=np.float64)
    Su = np.zeros((skipped_l,skipped_l),dtype=np.float64)
    Suu = np.zeros((skipped_l,skipped_l),dtype=np.float64)
    
    Rcurr=np.zeros(l,dtype=np.float64)
    Rprev=np.zeros(l,dtype=np.float64)
    Rprev2=np.zeros(l,dtype=np.float64)
    Scurr=np.zeros(l,dtype=np.float64)
    Sprev=np.zeros(l,dtype=np.float64)
    Sprev2=np.zeros(l,dtype=np.float64)
    Rprev[l-1]=R2_init[0]
    Sprev[l-1]=S2_init[0]
    if (l-1)%skip == 0:
        R[0,(l-1)//skip]=R2_init[0]
        S[0,(l-1)//skip]=S2_init[0]
    #R[i,l-i-1] = np.float64(R2_init[i])
    #S[i,l-i-1] = np.float64(S2_init[i])
    #if i + 1 < l:
    #    R[i+1,l-i-1] = np.float64(R_init[i])
    #    S[i+1,l-i-1] = np.float64(S_init[i])
    #equivalent to:
    #    R[i,l-i] = np.float64(R_init[i-1])
    #    S[i,l-i] = np.float64(S_init[i-1])
    
    for i in range(1,l):
        if DEBUG_PRINT and i % (l // 10) == 0:
            print(f"{i//(l//100)}% Done")
        #R[i,l-i-1] = np.float64(R2_init[i])
        #S[i,l-i-1] = np.float64(S2_init[i])
        Rcurr[l-i-1]=R2_init[i]
        Scurr[l-i-1]=S2_init[i]
        #R[i,l-i] = np.float64(R_init[i-1])
        #S[i,l-i] = np.float64(S_init[i-1])
        Rcurr[l-i] = R_init[i-1]
        Scurr[l-i] = S_init[i-1]
        for j in range(l-i+1,l):
            Q = (Qv[i]+Qv[i-1])/2 + (Quv[i]+Quv[i-1])/2*Quv_func(dv * (i-1/2) + du * (j-1/2)+delta_u_v)
            rstar = (dv * (i-1/2) + du * (j-1/2))/2 +delta_u_v_5
            if rstar != 0:
                rstar5 = rstar**(-5)
            else:
                rstar5 = 0
                
            pde_solve_point_sources5(Rcurr,Rprev, Scurr, Sprev, Q,dudv,i,j,(sources_r[i]+sources_r[i-1])/2,(sources_s[i]+sources_s[i-1])/2,
            (sources_rr[i]+sources_rr[i-1])/2,(sources_rs[i]+sources_rs[i-1])/2,(sources_r5[i]+sources_r5[i-1])/2*rstar5,(sources_s5[i]+sources_s5[i-1])/2*rstar5)
                
        if (i % skip) == 0:
            row = i // skip
            j = ((l-i-1)//skip) * skip
            for col in range((l-i-1)//skip,skipped_l):
                R[row,col]=Rcurr[j]
                S[row,col]=Scurr[j]
                if j > 0 and j+1<l:
                   Ru[row,col] = (Rcurr[j+1]-Rcurr[j-1]) / (2 * du)
                   Ruu[row,col] = (Rcurr[j+1]-2*Rcurr[j]+Rcurr[j-1]) / (du*du)
                   Su[row,col] = (Scurr[j+1]-Scurr[j-1]) / (2 * du)
                   Suu[row,col] = (Scurr[j+1]-2*Scurr[j]+Scurr[j-1]) / (du*du)
                j += skip
        if (i-1)%skip == 0:
            row = (i-1) // skip
            j = ((l-i-1)//skip) * skip
            for col in range((l-i-1)//skip,skipped_l):
               Rv[row,col] = (Rcurr[j]-Rprev2[j]) / (2 * dv)
               Rvv[row,col] = (Rcurr[j]-2*Rprev[j]+Rprev2[j]) / (dv*dv)
               Sv[row,col] = (Scurr[j]-Sprev2[j]) / (2 * dv)
               Svv[row,col] = (Scurr[j]-2*Sprev[j]+Sprev2[j]) / (dv*dv)
               j += skip
        Rcurr,Rprev,Rprev2 = Rprev2,Rcurr,Rprev
        Scurr,Sprev,Sprev2 = Sprev2,Scurr,Sprev
    if DEBUG_PRINT:
        print("Done calculating")
    return R,S,Rv,Rvv,Ru,Ruu,Sv,Svv,Su,Suu
                     
@njit
def pde_solve_Quv_scale_internal(R, S, length, skip, R_init, R2_init, S_init, S2_init, Qv, du, dv, sources_r, sources_s, Quv, delta_u_v,Quv_scale, sources_rr, sources_rs,sources_r5,sources_s5,delta_u_v_5):
    dudv = du * dv
    l=length
    skipped_l = (l-1)//skip+1
    # R derivatives
    Rv = np.zeros((skipped_l,skipped_l),dtype=np.float64)
    Rvv = np.zeros((skipped_l,skipped_l),dtype=np.float64)
    Ru = np.zeros((skipped_l,skipped_l),dtype=np.float64)
    Ruu = np.zeros((skipped_l,skipped_l),dtype=np.float64)
    # S derivatives
    Sv = np.zeros((skipped_l,skipped_l),dtype=np.float64)
    Svv = np.zeros((skipped_l,skipped_l),dtype=np.float64)
    Su = np.zeros((skipped_l,skipped_l),dtype=np.float64)
    Suu = np.zeros((skipped_l,skipped_l),dtype=np.float64)
    
    Rcurr=np.zeros(l,dtype=np.float64)
    Rprev=np.zeros(l,dtype=np.float64)
    Rprev2=np.zeros(l,dtype=np.float64)
    Scurr=np.zeros(l,dtype=np.float64)
    Sprev=np.zeros(l,dtype=np.float64)
    Sprev2=np.zeros(l,dtype=np.float64)
    Rprev[l-1]=R2_init[0]
    Sprev[l-1]=S2_init[0]
    if (l-1)%skip == 0:
        R[0,(l-1)//skip]=R2_init[0]
        S[0,(l-1)//skip]=S2_init[0]
    #R[i,l-i-1] = np.float64(R2_init[i])
    #S[i,l-i-1] = np.float64(S2_init[i])
    #if i + 1 < l:
    #    R[i+1,l-i-1] = np.float64(R_init[i])
    #    S[i+1,l-i-1] = np.float64(S_init[i])
    #equivalent to:
    #    R[i,l-i] = np.float64(R_init[i-1])
    #    S[i,l-i] = np.float64(S_init[i-1])
    
    for i in range(1,l):
        if DEBUG_PRINT and i % (l // 10) == 0:
            print(f"{i//(l//100)}% Done")
        #R[i,l-i-1] = np.float64(R2_init[i])
        #S[i,l-i-1] = np.float64(S2_init[i])
        Rcurr[l-i-1]=R2_init[i]
        Scurr[l-i-1]=S2_init[i]
        #R[i,l-i] = np.float64(R_init[i-1])
        #S[i,l-i] = np.float64(S_init[i-1])
        Rcurr[l-i] = R_init[i-1]
        Scurr[l-i] = S_init[i-1]
        for j in range(l-i+1,l):
            Q = (Qv[i]+Qv[i-1])/2 + (Quv[i]+Quv[i-1])/2*Quv_func((dv * (i-1/2) + du * (j-1/2)+delta_u_v)*(Quv_scale[i]+Quv_scale[i-1])/2)
            rstar = (dv * (i-1/2) + du * (j-1/2))/2 +delta_u_v_5
            if rstar != 0:
                rstar5 = rstar**(-5)
            else:
                rstar5 = 0
                
            pde_solve_point_sources5(Rcurr,Rprev, Scurr, Sprev, Q,dudv,i,j,(sources_r[i]+sources_r[i-1])/2,(sources_s[i]+sources_s[i-1])/2,
            (sources_rr[i]+sources_rr[i-1])/2,(sources_rs[i]+sources_rs[i-1])/2,(sources_r5[i]+sources_r5[i-1])/2*rstar5,(sources_s5[i]+sources_s5[i-1])/2*rstar5)
                
        if (i % skip) == 0:
            row = i // skip
            j = ((l-i-1)//skip) * skip
            for col in range((l-i-1)//skip,skipped_l):
                R[row,col]=Rcurr[j]
                S[row,col]=Scurr[j]
                if j > 0 and j+1<l:
                   Ru[row,col] = (Rcurr[j+1]-Rcurr[j-1]) / (2 * du)
                   Ruu[row,col] = (Rcurr[j+1]-2*Rcurr[j]+Rcurr[j-1]) / (du*du)
                   Su[row,col] = (Scurr[j+1]-Scurr[j-1]) / (2 * du)
                   Suu[row,col] = (Scurr[j+1]-2*Scurr[j]+Scurr[j-1]) / (du*du)
                j += skip
        if (i-1)%skip == 0:
            row = (i-1) // skip
            j = ((l-i-1)//skip) * skip
            for col in range((l-i-1)//skip,skipped_l):
               Rv[row,col] = (Rcurr[j]-Rprev2[j]) / (2 * dv)
               Rvv[row,col] = (Rcurr[j]-2*Rprev[j]+Rprev2[j]) / (dv*dv)
               Sv[row,col] = (Scurr[j]-Sprev2[j]) / (2 * dv)
               Svv[row,col] = (Scurr[j]-2*Sprev[j]+Sprev2[j]) / (dv*dv)
               j += skip
        Rcurr,Rprev,Rprev2 = Rprev2,Rcurr,Rprev
        Scurr,Sprev,Sprev2 = Sprev2,Scurr,Sprev
    if DEBUG_PRINT:
        print("Done calculating")
    return R,S,Rv,Rvv,Ru,Ruu,Sv,Svv,Su,Suu
    
def pde_solve(Ru_init, Rv_init, Su_init, Sv_init, Q, du, dv):
    R = np.zeros((len(Ru_init),len(Rv_init)),dtype=np.float64)
    R[0,:] = np.array(Rv_init)
    R[:,0] = np.array(Ru_init)
    
    S = np.zeros((len(Su_init),len(Sv_init)),dtype=np.float64)
    S[0,:] = np.array(Sv_init)
    S[:,0] = np.array(Su_init)
    return pde_solve_internal(R, S, np.float64(Q), np.float64(du), np.float64(dv))
    
def pde_solve_diagonal_with_derivatives(R_init, DR_init, S_init, DS_init, Q, du, dv):
    try: #Indexes order is R[v,u]
        l = len(R_init)
        if l != len(S_init):
            raise Exception("Inconsistent length of S_init and R_init", len(S_init), len(R_init))
            
        R = np.zeros((l,l),dtype=np.float64)
        S = np.zeros((l,l),dtype=np.float64)
        for i in range(l):
            R[i,l-i-1] = np.float64(R_init[i])
            S[i,l-i-1] = np.float64(S_init[i])
            if i < l - 1:
                R[i,l-i-2] = R[i,l-i-1]-DR_init[i]*du
                S[i,l-i-2] = S[i,l-i-1]-DS_init[i]*du
        return pde_solve_diagonal_internal(R, S, np.float64(Q), np.float64(du), np.float64(dv))
    except Exception:
        return traceback.format_exc()

def pde_solve_diagonal(R_init, R2_init, S_init, S2_init, Q, du, dv):
    try: #Indexes order is R[v,u]
        l = len(R_init)
        if l != len(S_init):
            raise Exception("Inconsistent length of S_init and R_init", len(S_init), len(R_init))
            
        R = np.zeros((l,l),dtype=np.float64)
        S = np.zeros((l,l),dtype=np.float64)
        for i in range(l):
            R[i,l-i-1] = np.float64(R_init[i])
            S[i,l-i-1] = np.float64(S_init[i])
            if i < l - 1:
                R[i,l-i-2] = np.float64(R2_init[i])
                S[i,l-i-2] = np.float64(S2_init[i])
        return pde_solve_diagonal_internal(R, S, np.float64(Q), np.float64(du), np.float64(dv))
    except Exception:
        return traceback.format_exc()
        
def pde_solve_diagonal2(R_init, R2_init, S_init, S2_init, Q, du, dv):
    try: #Indexes order is R[v,u]
        l = len(R2_init)
        if l != len(S2_init):
            raise Exception("Inconsistent length of S_init and R_init", len(S_init), len(R_init))
            
        R = np.zeros((l,l),dtype=np.float64)
        S = np.zeros((l,l),dtype=np.float64)
        for i in range(l):
            R[i,l-i-1] = np.float64(R2_init[i])
            S[i,l-i-1] = np.float64(S2_init[i])
            if i + 1 < l:
                R[i+1,l-i-1] = np.float64(R_init[i])
                S[i+1,l-i-1] = np.float64(S_init[i])
        return pde_solve_diagonal2_internal(R, S, np.float64(Q), np.float64(du), np.float64(dv))
    except Exception:
        return traceback.format_exc()
        
def pde_solve_with_fluxes(R_init, R2_init, S_init, S2_init, Q, du, dv, flux_r, flux_s):
    try: #Indexes order is R[v,u]
        l = len(R2_init)
        if l != len(S2_init):
            raise Exception("Inconsistent length of S_init and R_init", len(S_init), len(R_init))
            
        R = np.zeros((l,l),dtype=np.float64)
        S = np.zeros((l,l),dtype=np.float64)
        for i in range(l):
            R[i,l-i-1] = np.float64(R2_init[i])
            S[i,l-i-1] = np.float64(S2_init[i])
            if i + 1 < l:
                R[i+1,l-i-1] = np.float64(R_init[i])
                S[i+1,l-i-1] = np.float64(S_init[i])
        return pde_solve_with_fluxes_internal(R, S, np.float64(Q), np.float64(du), np.float64(dv),np.float64(flux_r),np.float64(flux_s))
    except Exception:
        return traceback.format_exc()
        
def pde_solve_with_sources(R_init, R2_init, S_init, S2_init, Q, du, dv, sources_r, sources_s):
    try: #Indexes order is R[v,u]
        l = len(R2_init)
        if l != len(S2_init):
            raise Exception("Inconsistent length of S_init and R_init", len(S_init), len(R_init))
            
        R = np.zeros((l,l),dtype=np.float64)
        S = np.zeros((l,l),dtype=np.float64)
        sources_r_array = np.zeros(l,dtype=np.float64)
        sources_s_array = np.zeros(l,dtype=np.float64)
        for i in range(l):
            sources_r_array[i]=np.float64(sources_r[i])
            sources_s_array[i]=np.float64(sources_s[i])
            R[i,l-i-1] = np.float64(R2_init[i])
            S[i,l-i-1] = np.float64(S2_init[i])
            if i + 1 < l:
                R[i+1,l-i-1] = np.float64(R_init[i])
                S[i+1,l-i-1] = np.float64(S_init[i])
        return pde_solve_with_sources_internal(R, S, np.float64(Q), np.float64(du), np.float64(dv),sources_r_array,sources_s_array)
    except Exception:
        return traceback.format_exc()
        
def pde_solve_with_charge(R_init, R2_init, S_init, S2_init, Qv, du, dv, sources_r, sources_s):
    try: #Indexes order is R[v,u]
        l = len(R2_init)
        if l != len(S2_init):
            raise Exception("Inconsistent length of S_init and R_init", len(S_init), len(R_init))
            
        R = np.zeros((l,l),dtype=np.float64)
        S = np.zeros((l,l),dtype=np.float64)
        sources_r_array = np.zeros(l,dtype=np.float64)
        sources_s_array = np.zeros(l,dtype=np.float64)
        Qarray = np.zeros(l,dtype=np.float64)
        for i in range(l):
            sources_r_array[i]=np.float64(sources_r[i])
            sources_s_array[i]=np.float64(sources_s[i])
            Qarray[i]=np.float64(Qv[i])
            R[i,l-i-1] = np.float64(R2_init[i])
            S[i,l-i-1] = np.float64(S2_init[i])
            if i + 1 < l:
                R[i+1,l-i-1] = np.float64(R_init[i])
                S[i+1,l-i-1] = np.float64(S_init[i])
        return pde_solve_with_charge_internal(R, S, Qarray, np.float64(du), np.float64(dv),sources_r_array,sources_s_array)
    except Exception:
        return traceback.format_exc()
        
def pde_solve_with_charge_Qr(R_init, R2_init, S_init, S2_init, Qv, du, dv, sources_r, sources_s, Qr, Qr0,delta_u_v):
    try: #Indexes order is R[v,u]
        l = len(R2_init)
        if l != len(S2_init):
            raise Exception("Inconsistent length of S_init and R_init", len(S_init), len(R_init))
            
        R = np.zeros((l,l),dtype=np.float64)
        S = np.zeros((l,l),dtype=np.float64)
        sources_r_array = np.zeros(l,dtype=np.float64)
        sources_s_array = np.zeros(l,dtype=np.float64)
        Qr_array = np.zeros(l,dtype=np.float64)
        Qr0_array = np.zeros(l,dtype=np.float64)
        Qarray = np.zeros(l,dtype=np.float64)
        for i in range(l):
            sources_r_array[i]=np.float64(sources_r[i])
            sources_s_array[i]=np.float64(sources_s[i])
            Qarray[i]=np.float64(Qv[i])
            Qr_array[i]=np.float64(Qr[i])
            Qr0_array[i]=np.float64(Qr0[i])
            R[i,l-i-1] = np.float64(R2_init[i])
            S[i,l-i-1] = np.float64(S2_init[i])
            if i + 1 < l:
                R[i+1,l-i-1] = np.float64(R_init[i])
                S[i+1,l-i-1] = np.float64(S_init[i])
        return pde_solve_with_charge_Qr_internal(R, S, Qarray, np.float64(du), np.float64(dv),sources_r_array,sources_s_array, Qr_array, Qr0_array,np.float64(delta_u_v))
    except Exception:
        return traceback.format_exc()
        
def pde_solve_with_charge_Quv(R_init, R2_init, S_init, S2_init, Qv, du, dv, sources_r, sources_s, Quv,delta_u_v):
    try: #Indexes order is R[v,u]
        l = len(R2_init)
        if l != len(S2_init):
            raise Exception("Inconsistent length of S_init and R_init", len(S_init), len(R_init))
            
        R = np.zeros((l,l),dtype=np.float64)
        S = np.zeros((l,l),dtype=np.float64)
        sources_r_array = np.zeros(l,dtype=np.float64)
        sources_s_array = np.zeros(l,dtype=np.float64)
        Quv_array = np.zeros(l,dtype=np.float64)
        Qarray = np.zeros(l,dtype=np.float64)
        for i in range(l):
            sources_r_array[i]=np.float64(sources_r[i])
            sources_s_array[i]=np.float64(sources_s[i])
            Qarray[i]=np.float64(Qv[i])
            Quv_array[i]=np.float64(Quv[i])
            R[i,l-i-1] = np.float64(R2_init[i])
            S[i,l-i-1] = np.float64(S2_init[i])
            if i + 1 < l:
                R[i+1,l-i-1] = np.float64(R_init[i])
                S[i+1,l-i-1] = np.float64(S_init[i])
        return pde_solve_with_charge_Quv_internal(R, S, Qarray, np.float64(du), np.float64(dv),sources_r_array,sources_s_array, Quv_array,np.float64(delta_u_v))
    except Exception:
        return traceback.format_exc()

def pde_solve_with_sources_r(R_init, R2_init, S_init, S2_init, Qv, du, dv, sources_r, sources_s, Quv,delta_u_v,sources_rr,sources_rs):
    try: #Indexes order is R[v,u]
        l = len(R2_init)
        if l != len(S2_init):
            raise Exception("Inconsistent length of S_init and R_init", len(S_init), len(R_init))
            
        R = np.zeros((l,l),dtype=np.float64)
        S = np.zeros((l,l),dtype=np.float64)
        sources_r_array = np.zeros(l,dtype=np.float64)
        sources_s_array = np.zeros(l,dtype=np.float64)
        sources_rr_array = np.zeros(l,dtype=np.float64)
        sources_rs_array = np.zeros(l,dtype=np.float64)
        Quv_array = np.zeros(l,dtype=np.float64)
        Qarray = np.zeros(l,dtype=np.float64)
        for i in range(l):
            sources_r_array[i]=np.float64(sources_r[i])
            sources_s_array[i]=np.float64(sources_s[i])
            sources_rr_array[i]=np.float64(sources_rr[i])
            sources_rs_array[i]=np.float64(sources_rs[i])
            Qarray[i]=np.float64(Qv[i])
            Quv_array[i]=np.float64(Quv[i])
            R[i,l-i-1] = np.float64(R2_init[i])
            S[i,l-i-1] = np.float64(S2_init[i])
            if i + 1 < l:
                R[i+1,l-i-1] = np.float64(R_init[i])
                S[i+1,l-i-1] = np.float64(S_init[i])
        return pde_solve_with_sources_r_internal(R, S, Qarray, np.float64(du), np.float64(dv),sources_r_array,sources_s_array, 
        Quv_array,np.float64(delta_u_v),sources_rr_array,sources_rs_array)
    except Exception:
        return traceback.format_exc()

def pde_solve_skip(skip, R_init, R2_init, S_init, S2_init, Qv, du, dv, sources_r, sources_s, Quv,delta_u_v,sources_rr,sources_rs):
    try: #Indexes order is R[v,u]
        l = len(R2_init)
        if l != len(S2_init):
            raise Exception("Inconsistent length of S_init and R_init", len(S_init), len(R_init))
        
        skipped_l = (l-1)//skip+1
        R = np.zeros((skipped_l,skipped_l),dtype=np.float64)
        S = np.zeros((skipped_l,skipped_l),dtype=np.float64)

        R_init_array = np.zeros(l,dtype=np.float64)
        R2_init_array = np.zeros(l,dtype=np.float64)
        S_init_array = np.zeros(l,dtype=np.float64)
        S2_init_array = np.zeros(l,dtype=np.float64)
        sources_r_array = np.zeros(l,dtype=np.float64)
        sources_s_array = np.zeros(l,dtype=np.float64)
        sources_rr_array = np.zeros(l,dtype=np.float64)
        sources_rs_array = np.zeros(l,dtype=np.float64)
        Quv_array = np.zeros(l,dtype=np.float64)
        Qarray = np.zeros(l,dtype=np.float64)
        for i in range(l):
            sources_r_array[i]=np.float64(sources_r[i])
            sources_s_array[i]=np.float64(sources_s[i])
            sources_rr_array[i]=np.float64(sources_rr[i])
            sources_rs_array[i]=np.float64(sources_rs[i])
            Qarray[i]=np.float64(Qv[i])
            Quv_array[i]=np.float64(Quv[i])
            if i + 1 < l:
                R_init_array[i]=np.float64(R_init[i])
                S_init_array[i]=np.float64(S_init[i])
            R2_init_array[i]=np.float64(R2_init[i])
            S2_init_array[i]=np.float64(S2_init[i])
            
        return pde_solve_skip_internal(R, S, l, skip, R_init_array, R2_init_array, S_init_array, S2_init_array, Qarray, np.float64(du), np.float64(dv),sources_r_array,sources_s_array, 
        Quv_array,np.float64(delta_u_v),sources_rr_array,sources_rs_array)
    except Exception:
        return traceback.format_exc()
        
def pde_solve_sources5(skip, R_init, R2_init, S_init, S2_init, Qv, du, dv, sources_r, sources_s, Quv,delta_u_v,sources_rr,sources_rs,sources_r5,sources_s5,delta_u_v_5):
    try: #Indexes order is R[v,u]
        l = len(R2_init)
        if l != len(S2_init):
            raise Exception("Inconsistent length of S_init and R_init", len(S_init), len(R_init))
        
        skipped_l = (l-1)//skip+1
        R = np.zeros((skipped_l,skipped_l),dtype=np.float64)
        S = np.zeros((skipped_l,skipped_l),dtype=np.float64)

        R_init_array = np.zeros(l,dtype=np.float64)
        R2_init_array = np.zeros(l,dtype=np.float64)
        S_init_array = np.zeros(l,dtype=np.float64)
        S2_init_array = np.zeros(l,dtype=np.float64)
        sources_r_array = np.zeros(l,dtype=np.float64)
        sources_s_array = np.zeros(l,dtype=np.float64)
        sources_rr_array = np.zeros(l,dtype=np.float64)
        sources_rs_array = np.zeros(l,dtype=np.float64)
        sources_r5_array = np.zeros(l,dtype=np.float64)
        sources_s5_array = np.zeros(l,dtype=np.float64)
        Quv_array = np.zeros(l,dtype=np.float64)
        Qarray = np.zeros(l,dtype=np.float64)
        for i in range(l):
            sources_r_array[i]=np.float64(sources_r[i])
            sources_s_array[i]=np.float64(sources_s[i])
            sources_rr_array[i]=np.float64(sources_rr[i])
            sources_rs_array[i]=np.float64(sources_rs[i])
            sources_r5_array[i]=np.float64(sources_r5[i])
            sources_s5_array[i]=np.float64(sources_s5[i])
            Qarray[i]=np.float64(Qv[i])
            Quv_array[i]=np.float64(Quv[i])
            if i + 1 < l:
                R_init_array[i]=np.float64(R_init[i])
                S_init_array[i]=np.float64(S_init[i])
            R2_init_array[i]=np.float64(R2_init[i])
            S2_init_array[i]=np.float64(S2_init[i])
            
        return pde_solve_sources5_internal(R, S, l, skip, R_init_array, R2_init_array, S_init_array, S2_init_array, Qarray, np.float64(du), np.float64(dv),sources_r_array,sources_s_array,
        Quv_array,np.float64(delta_u_v),sources_rr_array,sources_rs_array,sources_r5_array,sources_s5_array,np.float64(delta_u_v_5))
    except Exception:
        return traceback.format_exc()
        
def pde_solve_Quv_scale(skip, R_init, R2_init, S_init, S2_init, Qv, du, dv, sources_r, sources_s, Quv,delta_u_v,Quv_scale,sources_rr,sources_rs,sources_r5,sources_s5,delta_u_v_5):
    try: #Indexes order is R[v,u]
        l = len(R2_init)
        if l != len(S2_init):
            raise Exception("Inconsistent length of S_init and R_init", len(S_init), len(R_init))
        
        skipped_l = (l-1)//skip+1
        R = np.zeros((skipped_l,skipped_l),dtype=np.float64)
        S = np.zeros((skipped_l,skipped_l),dtype=np.float64)

        R_init_array = np.zeros(l,dtype=np.float64)
        R2_init_array = np.zeros(l,dtype=np.float64)
        S_init_array = np.zeros(l,dtype=np.float64)
        S2_init_array = np.zeros(l,dtype=np.float64)
        sources_r_array = np.zeros(l,dtype=np.float64)
        sources_s_array = np.zeros(l,dtype=np.float64)
        sources_rr_array = np.zeros(l,dtype=np.float64)
        sources_rs_array = np.zeros(l,dtype=np.float64)
        sources_r5_array = np.zeros(l,dtype=np.float64)
        sources_s5_array = np.zeros(l,dtype=np.float64)
        Quv_array = np.zeros(l,dtype=np.float64)
        Quv_scale_array = np.zeros(l,dtype=np.float64)
        Qarray = np.zeros(l,dtype=np.float64)
        for i in range(l):
            sources_r_array[i]=np.float64(sources_r[i])
            sources_s_array[i]=np.float64(sources_s[i])
            sources_rr_array[i]=np.float64(sources_rr[i])
            sources_rs_array[i]=np.float64(sources_rs[i])
            sources_r5_array[i]=np.float64(sources_r5[i])
            sources_s5_array[i]=np.float64(sources_s5[i])
            Qarray[i]=np.float64(Qv[i])
            Quv_array[i]=np.float64(Quv[i])
            Quv_scale_array[i]=np.float64(Quv_scale[i])
            if i + 1 < l:
                R_init_array[i]=np.float64(R_init[i])
                S_init_array[i]=np.float64(S_init[i])
            R2_init_array[i]=np.float64(R2_init[i])
            S2_init_array[i]=np.float64(S2_init[i])
            
        return pde_solve_Quv_scale_internal(R, S, l, skip, R_init_array, R2_init_array, S_init_array, S2_init_array, Qarray, np.float64(du), np.float64(dv),sources_r_array,sources_s_array,
        Quv_array,np.float64(delta_u_v),Quv_scale_array,sources_rr_array,sources_rs_array,sources_r5_array,sources_s5_array,np.float64(delta_u_v_5))
    except Exception:
        return traceback.format_exc()