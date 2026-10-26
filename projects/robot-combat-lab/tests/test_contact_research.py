import json,numpy as np,mujoco,torch
from combatlab.contact_arena import ContactArena
from combatlab.contact_training import ContactPolicy
from combatlab.research_archive import workspace,digest

def test_native_self_collision_relations_and_cross_contact_tags():
 env=ContactArena('k1','go2')
 try:
  bybody={}
  for i in range(env.m.ngeom):
   b=env.m.body(int(env.m.geom_bodyid[i])).name or ''
   if b.startswith('p0/'):bybody.setdefault(b,[]).append(i)
  ids=[i for v in bybody.values() for i in v if env.m.geom_contype[i]]
  for a in ids:
   for b in ids:
    assert not ((env.m.geom_contype[a]&env.m.geom_conaffinity[b]) or (env.m.geom_contype[b]&env.m.geom_conaffinity[a]))
  other=next(i for i in range(env.m.ngeom) if (env.m.body(int(env.m.geom_bodyid[i])).name or '').startswith('p1/') and env.m.geom_contype[i])
  assert all((env.m.geom_contype[i]&env.m.geom_conaffinity[other]) or (env.m.geom_contype[other]&env.m.geom_conaffinity[i]) for i in ids)
  for _ in range(20):env.step(np.ones((2,32))*.3)
  assert not env.d.xfrc_applied.any() and not env.d.qfrc_applied.any()
  for actor in env.actors:assert np.all(np.abs(env.d.actuator_force[actor.actuator_ids])<=actor.torque_limits+1e-7)
 finally:env.close()

def test_ppo_inactive_action_dimensions_do_not_change_probability_ratio():
 torch.manual_seed(4);p=ContactPolicy();x=torch.randn(2,131);dist=p.distribution(x);a=dist.sample();mask=torch.zeros(2,32);mask[:,:15]=1;first=(dist.log_prob(a)*mask).sum(-1);a[:,15:]+=100;second=(dist.log_prob(a)*mask).sum(-1);torch.testing.assert_close(first,second)

def test_preserved_baseline_inventory_and_checkpoint_rng_fields():
 root=workspace();summary=json.loads((root/'08_manifests/baseline-summary.json').read_text());assert summary['state']=='complete' and summary['verified_files']>=20000
 assert digest(root/'08_manifests/baseline-files.jsonl')==summary['inventory_sha256']
 for line in (root/'08_manifests/baseline-files.jsonl').read_text().splitlines():
  row=json.loads(line)
  if row['path'] in ['src/combatlab/duel.py','results/duel-index.json']:
   assert digest(root/'01_baseline/project'/row['path'])==row['sha256']
 p=root/'03_training/ppo-go2-20261025T123100-e5e788/checkpoints/iteration-00024.pt';c=torch.load(p,map_location='cpu',weights_only=False)
 assert {'policy','optimizer','torch_rng','numpy_rng','environment_rng','config','next_episode'}<=c.keys()
