"""Regression checks for archive corruption and actuator identity, not game scores."""
import json
import numpy as np
import pytest
from combatlab.storage import encode_model,load_binary,load_model
from combatlab.models import ROOT,make_actors


def test_exact_model_patch_roundtrip_and_tamper(tmp_path):
    a=tmp_path/'results/matches/a';b=tmp_path/'results/matches/b'
    a.mkdir(parents=True);b.mkdir()
    original=bytes(range(256))*8
    changed=bytearray(original);changed[10:14]=b'ABCD';changed[-1]=17
    encode_model(a,original,('test',));meta=encode_model(b,bytes(changed),('test',))
    assert meta['changed_bytes']==5
    assert load_binary(a)==original and load_binary(b)==bytes(changed)
    patch=b/'model.patch.json.gz';patch.write_bytes(patch.read_bytes()+b'corruption')
    with pytest.raises(AssertionError):load_binary(b)


def test_h1_nontrivial_actuator_order_preserves_joint_identity():
    folder=ROOT/'results/matches/robot-03-f2-m0'
    r=json.loads((folder/'result.json').read_text());m=load_model(folder)
    actors=make_actors(m,tuple(r['robots']))
    for a in actors:
        assert np.array_equal(m.jnt_qposadr[m.actuator_trnid[a.actuator_ids,0]],a.qadr)
        assert np.array_equal(m.jnt_dofadr[m.actuator_trnid[a.actuator_ids,0]],a.vadr)
        assert np.all(a.torque_limits>0)
        assert np.allclose(m.actuator_forcerange[a.actuator_ids,1],a.torque_limits)


def test_rank_verification_does_not_mutate_published_dataset():
    from combatlab.benchmark import rank
    path=ROOT/'results/leaderboards.json';before=path.read_bytes()
    expected=json.loads(before);actual=rank(expected['matches'],persist=False)
    assert actual['leaderboards']==expected['leaderboards']
    assert path.read_bytes()==before
