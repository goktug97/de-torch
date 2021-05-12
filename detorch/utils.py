from functools import wraps


def hook(func):
    func.before_register = []
    func.after_register = []
    def add_hook(after=True):
        def _add_hook(hook):
            if after:
                func.after_register.append(hook)
            else:
                func.before_register.append(hook)
        return _add_hook

    func.add_hook = add_hook

    @wraps(func)
    def wrapped(*args, **kwargs):
        for hook in func.before_register:
            hook(*args, **kwargs)
        ret = func(*args, **kwargs)
        for hook in func.after_register:
            hook(*args, **kwargs, ret=ret)
        return ret

    return wrapped
