from utils.utils import *
from core.natural_gradient_ray import local_conjugate_gradient_parallel
from trpo_server import TRPOServer

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
torch.set_default_tensor_type('torch.DoubleTensor')


class LocalTRPOServer(TRPOServer):
    def __init__(self, args, dtype=torch.double):
        super(LocalTRPOServer, self).__init__(args, dtype)

    def conjugate_gradient(self, actor_gradient_list, states_list):
        conjugate_gradient_directions = local_conjugate_gradient_parallel(
            policy_net=self.actor,
            states_list=states_list,
            pg_list=actor_gradient_list,
            max_kl=self.args.max_kl,
            cg_damping=self.args.cg_damping,
            cg_iter=self.args.cg_iter
        )
        conjugate_gradient_direction = np.array(conjugate_gradient_directions).mean(axis=0)
        conjugate_gradient_direction = torch.from_numpy(conjugate_gradient_direction)
        return conjugate_gradient_direction
