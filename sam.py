from fastai.basics import *
import copy

class SAM(Callback):
    def __init__(self,rho=.05):
        self.rho = rho
    
    def before_fit(self):
        self.learn.rho = self.rho
        
        # overwriting _do_one_batch to use SAM
        @patch
        def _do_one_batch(self:Learner):
            
            #The SAM operation
            def virtual_batch(xb,yb):
                #Making a seperate copy of model to not disturb batchnorm
                tmp_model = copy.deepcopy(self.model)
                with torch.no_grad():
                    #taking virtual step on the model copy
                    for p,q in zip(trainable_params(tmp_model),trainable_params(self.model)):
                        p.data += -self.rho * q.grad.data/q.grad.data.norm(2)
                        q.grad.detach_()
                        q.grad.zero_()
                #Evaluating the batch using the model copy
                pred = tmp_model(*xb)
                loss_grad = self.loss_func(pred, *yb)
                loss_grad.backward()
                #pass the gradients back to the main model
                for p,q in zip(trainable_params(tmp_model),trainable_params(self.model)):
                    q.grad.data += p.grad.data
                  
            #Begin the Batch
            xb = self.xb
            yb = self.yb
            self.pred = self.model(*xb)
            self('after_pred')
            if len(yb):
                self.loss_grad = self.loss_func(self.pred, *yb)
                self.loss = self.loss_grad.clone()
            self('after_loss')
            if not self.training or not len(yb): return
            self('before_backward')
            self.loss_grad.backward()
            virtual_batch(xb,yb)            
            self._with_events(self.opt.step, 'step', CancelStepException)
            self.opt.zero_grad()
    
    # Restore normal behavior on Cleanup
    @patch
    def _end_cleanup(self:Learner):
        self.dl,self.xb,self.yb,self.pred,self.loss = None,(None,),(None,),None,None
        
        @patch
        def _do_one_batch(self:Learner):
            self.pred = self.model(*self.xb)
            self('after_pred')
            if len(self.yb):
                self.loss_grad = self.loss_func(self.pred, *self.yb)
                self.loss = self.loss_grad.clone()
            self('after_loss')
            if not self.training or not len(self.yb): return
            self('before_backward')
            self.loss_grad.backward()
            self._with_events(self.opt.step, 'step', CancelStepException)
            self.opt.zero_grad()
