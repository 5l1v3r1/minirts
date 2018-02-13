class RTSEnv(object):
    def seed(self):
    	raise NotImplementedError('Your sublass RTS should override this!')
    def reset(self):
    	raise NotImplementedError('Your sublass RTS should override this!')
    def step(self):
    	raise NotImplementedError('Your sublass RTS should override this!')
    def render(self):
    	raise NotImplementedError('Your sublass RTS should override this!')