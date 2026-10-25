"""Optional parameterized motor model. Missing hardware data stays UNKNOWN.

This is a lumped thermal model, not a material/gearbox fracture predictor.
No thresholds are inferred from an actuator's peak torque specification.
"""
from dataclasses import dataclass,asdict
import math
import numpy as np

@dataclass(frozen=True)
class ThermalParameters:
    loss_w_per_nm2: float
    capacity_j_per_k: float
    conductance_w_per_k: float
    ambient_c: float
    initial_c: float
    trip_c: float
    reset_c: float
    irreversible_c: float
    irreversible_hold_s: float
    provenance: str
    evidence_status: str

    def __post_init__(self):
        values=[v for v in asdict(self).values() if isinstance(v,(float,int))]
        if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite thermal parameter')
        if self.loss_w_per_nm2<0 or self.capacity_j_per_k<=0 or self.conductance_w_per_k<0:raise ValueError('Invalid thermal coefficients')
        if self.reset_c>=self.trip_c or self.irreversible_hold_s<=0:raise ValueError('Invalid protection thresholds')
        if not self.provenance.strip() or self.evidence_status not in ['hypothetical','measured']:raise ValueError('Explicit source/evidence status required')

class HardwareMonitor:
    def __init__(self,model,actors,parameters=None):
        self.model=model;self.actors=actors;self.base_limits=model.actuator_forcerange.copy();self.parameters=parameters or {};self.names=[model.actuator(i).name for i in range(model.nu)];self.temperature=np.full(model.nu,np.nan);self.tripped=np.zeros(model.nu,dtype=bool);self.burned=np.zeros(model.nu,dtype=bool);self.overheat_s=np.zeros(model.nu);self.events=[]
        unknown=set(self.parameters)-set(self.names)
        if unknown:raise ValueError(f'Unknown actuator keys: {sorted(unknown)}')
        for i,name in enumerate(self.names):
            if name in self.parameters:self.temperature[i]=self.parameters[name].initial_c

    def step(self,torque,dt,time):
        for i,name in enumerate(self.names):
            p=self.parameters.get(name)
            if p is None:continue
            old=(bool(self.tripped[i]),bool(self.burned[i]));T=self.temperature[i]
            # Exact integration of C*dT/dt = alpha*tau^2 - h*(T-Ta)
            heat=p.loss_w_per_nm2*float(torque[i])**2
            if p.conductance_w_per_k:
                decay=math.exp(-p.conductance_w_per_k*dt/p.capacity_j_per_k);eq=p.ambient_c+heat/p.conductance_w_per_k;T=eq+(T-eq)*decay
            else:T+=heat*dt/p.capacity_j_per_k
            self.temperature[i]=T
            self.overheat_s[i]=self.overheat_s[i]+dt if T>=p.irreversible_c else 0.
            if self.overheat_s[i]>=p.irreversible_hold_s:self.burned[i]=True
            if T>=p.trip_c:self.tripped[i]=True
            elif T<=p.reset_c:self.tripped[i]=False
            if old!=(bool(self.tripped[i]),bool(self.burned[i])):self.events.append(dict(time=time,actuator=name,temperature_c=T,tripped=bool(self.tripped[i]),irreversible_model_fault=bool(self.burned[i]),evidence_status=p.evidence_status))
        self.apply()

    def apply(self):
        self.model.actuator_forcerange[:]=self.base_limits
        self.model.actuator_forcerange[self.tripped|self.burned]=0.

    def loss_of_all_actuation(self):
        return [bool(np.all(self.burned[a.actuator_ids])) for a in self.actors]

    def summary(self,side):
        ids=self.actors[side].actuator_ids;known=np.isfinite(self.temperature[ids]);temps=self.temperature[ids][known]
        return dict(mode='parameterized_unvalidated' if known.any() else 'not_calibrated',modeled_motors=int(known.sum()),total_motors=len(ids),max_temperature_c=float(temps.max()) if len(temps) else None,thermal_trips=int(self.tripped[ids].sum()),irreversible_motor_faults=int(self.burned[ids].sum()),structural_fracture='not_modeled',gearbox_failure='not_modeled',real_hardware_damage_confirmed=False)

    def state(self):
        return dict(base_limits=self.base_limits.tolist(),parameters={k:asdict(v) for k,v in self.parameters.items()},temperature_c=[float(x) if np.isfinite(x) else None for x in self.temperature],tripped=self.tripped.tolist(),burned=self.burned.tolist(),overheat_s=self.overheat_s.tolist(),events=self.events)

    def restore(self,state):
        self.base_limits[:]=state['base_limits'];self.temperature[:]=[np.nan if x is None else x for x in state['temperature_c']];self.tripped[:]=state['tripped'];self.burned[:]=state['burned'];self.overheat_s[:]=state['overheat_s'];self.events=state['events'];self.apply()
