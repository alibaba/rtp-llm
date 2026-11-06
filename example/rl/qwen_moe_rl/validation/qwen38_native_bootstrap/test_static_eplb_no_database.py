from pathlib import Path
import ast,types,unittest
HERE=Path(__file__).resolve().parent
path=HERE/'overlay/rtp_llm/model_loader/loader.py'
module=ast.parse(path.read_text());node=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='ModelLoader')
function=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='create_eplb')
class StaticPlacementTest(unittest.TestCase):
    def test_static_placement_never_opens_secondary_checkpoint_database(self):
        class LoadConfig:
            @staticmethod
            def create_redundant_expert(**kwargs):
                assert kwargs=={'layer_num':92,'expert_num':512,'ep_size':72,'num_nodes':18,'phy_exp_num':576,'phy2log_path':'frozen-physical-map.json'}
                return 'verified-fixed-placement'
        def reject(*args,**kwargs):raise AssertionError('checkpoint payload/database opened during static metadata bootstrap')
        scope={'LoadConfig':LoadConfig,'CkptDatabase':reject,'logging':types.SimpleNamespace(info=lambda *args:None)}
        exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),scope)
        model=types.SimpleNamespace(num_layers=92,expert_num=512,phy2log_path='frozen-physical-map.json',ckpt_path='/must/not/be/opened')
        weights=types.SimpleNamespace(expert_num_=512,phy_exp_num_=576,ep_size=72,num_nodes=18,enable_eplb_=False)
        instance=types.SimpleNamespace(model_config=model,_weights_info=weights)
        self.assertEqual(scope['create_eplb'](instance),(None,'verified-fixed-placement'))
if __name__=='__main__':unittest.main(verbosity=2)
