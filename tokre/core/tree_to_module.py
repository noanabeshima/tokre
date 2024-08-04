from tokre.core.modules import Toks, Repeat, Phrase, OrGroup, VarDefn, VarRef, Wildcard, LearnedConst, Lookbehind, Lookahead
from tokre.core.macros import DEFINED_MACROS
from tokre.core.parsing import parse
from lark import Transformer
import tiny_model



class InsertModules(Transformer):
    def repeat(self, children):
        assert len(children) == 3
        child_matcher, repeat_min, repeat_max = children
        return Repeat(child_matcher=child_matcher, min=repeat_min, max=repeat_max)

    def string(self, children):
        assert len(children) == 1, children
        assert isinstance(children[0], str)
        toks = tiny_model.raw_toks[
            tiny_model.enc(stories=children)[0].tolist()
        ].tolist()
        return Toks(toks=toks)

    def wildcard(self, children):
        return Wildcard()

    def phrase(self, children):
        return Phrase(matchers=children)

    def or_pattern(self, children):
        return OrGroup(matchers=children)

    def var_defn(self, children):
        assert len(children) == 2
        var_name, child_matcher = children
        return VarDefn(var_name=var_name, child_matcher=child_matcher)

    def var_ref(self, children):
        assert len(children) == 1, children
        var_name = children[0]
        return VarRef(var_name=var_name)

    def macro(self, children):
        assert len(children) >= 1

        macro_name, children = children[0], children[1:]

        args, kwargs = [], {}

        for child in children:
            if isinstance(child, dict):
                kwargs = kwargs | child
            else:
                args.append(child)

        if macro_name in DEFINED_MACROS:
            return DEFINED_MACROS[macro_name](*args, **kwargs)
        else:
            assert False, f"macro {macro_name} not found in macros.py"

def tree_to_module(tree):
    module = InsertModules().transform(tree)
    return module

from torch import nn
def recursively_add_name_to_submodule(module):
    assert not hasattr(module, 'name_to_submodule'), 'module already has name_to_submodule attribute'
    name_to_submodule = {}
    
    #[STUB] hard to read code
    def add_named_submodules(module, name_to_submodule):
        for submodule in module.children():
            if hasattr(submodule, 'name'):
                assert submodule.name not in name_to_submodule, 'Two tokre module children seem to have the same name?'
                name_to_submodule[submodule.name] = submodule
    
            if isinstance(submodule, nn.ModuleList):
                add_named_submodules(submodule, name_to_submodule)
        
        return name_to_submodule


    add_named_submodules(module, name_to_submodule)
    
    module.name_to_submodule = name_to_submodule
    for submodule in module.children():
        recursively_add_name_to_submodule(submodule)

def compile(s):
    tree = parse(s)
    module = tree_to_module(tree)
    if len(list(module.parameters())) == 0:
        module = LearnedConst(module)
    recursively_add_name_to_submodule(module)
    return module