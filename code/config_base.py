def get_type_full_name(t):
    assert isinstance(t, type)
    res = t.__qualname__
    if t.__module__ not in ('builtins', '__main__'):
        res = t.__module__ + '.' + res
    return res

class ConfigBase(object):

    def __init__(self, cfg, user_args, **kwargs):
        # We need this class to avoid passing
        # too many params in model classes initializers
        # Example usage:
        #    class MyConfig(ConfigBase):
        #        def __init__(self, **kwargs):
        #            super().__init__(self,
        #                             user_args = kwargs,
        #                             emb_size = None,
        #                             hid_size = None,
        #                             dropout_prob = None)
        #
        #    my_config = MyConfig(emb_size = 150,
        #                         hid_size = 50,
        #                         dropout_prob = 0)

        for key, user_val in user_args.items():
            assert key in kwargs, 'Unknown field "{}" with value "{}"'.format(key, user_val)
            assert user_val is not None, 'Field "{}" can\'t have "None" value'.format(key)

            default_val = kwargs[key]
            is_subconfig = isinstance(default_val, type) and issubclass(default_val, ConfigBase)

            if isinstance(user_val, ConfigBase):
                assert is_subconfig, \
                       'Field "{}" is not a sub-config and can\'t be set with config "{}"' \
                       .format(key, get_type_full_name(type(user_val)))
                assert type(user_val) == default_val, \
                       'Sub-config "{}" must have type "{}", instead "{}" given' \
                       .format(key, get_type_full_name(default_val), get_type_full_name(type(user_val)))
            elif is_subconfig:
                assert isinstance(user_val, dict), \
                       'Sub-config "{}" of type "{}" can only be created from dict, not "{}" of type "{}"' \
                       .format(key, get_type_full_name(default_val), user_val, get_type_full_name(type(user_val)))
                user_val = default_val(**user_val)

            kwargs[key] = user_val

        for key, user_val in kwargs.items():
            assert user_val is not None, 'Field "{}" without default value is not set'.format(key)
            setattr(cfg, key, user_val)

        self._keys = list(kwargs.keys())

    def as_dict(self):
        res = {key: getattr(self, key) for key in self._keys}
        for key in self._keys:
            if isinstance(res[key], ConfigBase):
                res[key] = res[key].as_dict()
        return res