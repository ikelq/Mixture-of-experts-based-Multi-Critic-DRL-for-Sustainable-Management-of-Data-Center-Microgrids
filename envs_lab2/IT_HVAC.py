import os
import numpy as np




class IT_HVAC():
    def __init__(self):
        super().__init__()

        # cpu 典型参数
        # Xeon 8468 (Sapphire Rapids)
        self.num_cpu = 250*40
        self.cpu_P_idle = 110
        self.cpu_TDP = 350
        self.cpu_T_base = 25
        self.cpu_alpha = 0.3
        # 典型参数（以 Xeon 风冷 1U 为例）
        self.cpu_fun_P_max = 45
        self.k0 = 25
        self.ku = 80
        self.kT = 3
        # self.cpu_fun_V_max = self.k0 + self.ku +self.kT*(self.RACK_SUPPLY_APPROACH_TEMP+ self.max_crac_temp - self.cpu_T_base)
        self.cpu_fun_V_max = 111



        # hvac 典型参数
        self.RACK_SUPPLY_APPROACH_TEMP = 8
        self.RACK_RETURN_APPROACH_TEMP = -4
        self.C_AIR = 1006
        self.RHO_AIR = 1.225
        self.CRAC_SUPPLY_AIR_FLOW_RATE_pu = 0.00005663
        self.CRAC_REFRENCE_AIR_FLOW_RATE_pu = 0.00009438
        self.CRAC_FAN_REF_P = 150
        self.CHILLER_COP = 6.0
        self.CT_FAN_REF_P = 1000
        self.CT_REFRENCE_AIR_FLOW_RATE = 2.8315
        self.CW_PRESSURE_DROP = 300000
        self.CW_WATER_FLOW_RATE = 0.0011
        self.CW_PUMP_EFFICIENCY = 0.87
        self.CT_PRESSURE_DROP = 300000
        self.CT_WATER_FLOW_RATE = 0.0011
        self.CT_PUMP_EFFICIENCY = 0.87

        self.a = 7.08
        self.b = 0.25
        # if ambient_temp - CRAC_setpoint > 0:
        # Coefficient of Performance
        # Time series of heat demand and heat pump efficiency for energy system modeling

        self.alpha_crac = 0.09
        self.gamma_crac = 0.02
        # α（基础损耗）	包括风扇、电机发热、建筑散热等不与温差直接相关的系数	0.05 ~ 0.15（即 5%~15%）	多篇指南推荐数据中心风机/结构损耗在此范围
        # γ（温差敏感系数）	制冷负载对环境温差的线性响应 表示温差每升高1°C导致冷负荷增长多 0.005 ~ 0.02 即每 °C 增加 0.5%~2%

# CRAC_temp_setpoint 15-22
# RACK_SUPPLY_APPROACH_TEMP = 5
    def IT_power_consumption(self, CRAC_temp_setpoint, cpu_load):

        cpu_inlet_temp = self.RACK_SUPPLY_APPROACH_TEMP + CRAC_temp_setpoint

        cpu_pwr = self.cpu_P_idle + cpu_load *(self.cpu_TDP - self.cpu_P_idle) + self.cpu_alpha* max(0,cpu_inlet_temp - self.cpu_T_base)**2

# 风量self.cpu_v_fan,是服务器风扇调速模型的核心变量，它反映了为了冷却当前 CPU负载 cpu_load和温度, 风扇需要提供的空气流量（单位：CFM）。典型参数（以 Xeon 风冷 1U 为例） k0=25 CFM, ku=80 CFM, kT=3 CFM /°C, self.cpu_T_base 58 °C 控温阈值，来自风扇 BIOS curve 起始点

        # cpu_temp = cpu_inlet_temp + cpu_pwr
        self.rack_v_fan = self.k0 + self.ku* cpu_load + self.kT*max(0,cpu_inlet_temp - self.cpu_T_base)
        P_cup_fan = self.cpu_fun_P_max * (self.rack_v_fan / self.cpu_fun_V_max)**3
        IT_power_per_cpu = cpu_pwr + P_cup_fan
        IT_power_consumption = self.num_cpu * IT_power_per_cpu
        delta_cpu_in_out_temp = IT_power_per_cpu/(self.C_AIR*self.RHO_AIR* self.rack_v_fan)
        # *0.75 why time 0.75?
        cpu_outlet_temp = cpu_inlet_temp + delta_cpu_in_out_temp
        CRAC_return_temp = cpu_outlet_temp + self.RACK_RETURN_APPROACH_TEMP

        return IT_power_consumption, self.num_cpu*cpu_pwr, self.num_cpu*P_cup_fan 
    

    def calculate_HVAC_power(self, CRAC_temp_setpoint, ambient_temp, IT_power_consumption):

        # if ambient_temp - CRAC_setpoint > 0:
        # Coefficient of Performance
        # Time series of heat demand and heat pump efficiency for energy system modeling
        cop = self.a - self.b* max (0, ambient_temp - CRAC_temp_setpoint) + 0.0005*max (0, ambient_temp - CRAC_temp_setpoint)**2

        # α（基础损耗）	包括风扇、电机发热、建筑散热等不与温差直接相关的系数	0.05 ~ 0.15（即 5%~15%）	多篇指南推荐数据中心风机/结构损耗在此范围
        # γ（温差敏感系数）	制冷负载对环境温差的线性响应 表示温差每升高1°C导致冷负荷增长多 0.005 ~ 0.02 即每 °C 增加 0.5%~2%

        cooling_load = IT_power_consumption*(1 + self.alpha_crac + self.gamma_crac*(ambient_temp - CRAC_temp_setpoint))

        # P_chiller_fan_max = 100e3
        
        # 归一化温差： Δ𝑇用于反映冷凝温差
        # P_chiller_fan = P_chiller_fan_max*((max(0, ambient_temp - CRAC_setpoint))/ 11)**3 * ((cooling_load )/ 2e6)**3
        P_chiller_fan =0
         
        chiller_power_consumption = cooling_load/cop
        # else:
        #     chiller_power_consumption = 0    # feeling cooling model, chiller not work
        
        # CRAC_fan_power_consumption =  IT_power_consumption/ (self.C_AIR*self.RHO_AIR(CRAC_return_temp- CRAC_setpoint))
        # HVAC_power_consumption = chiller_power_consumption + CRAC_fan_power_consumption

        return chiller_power_consumption