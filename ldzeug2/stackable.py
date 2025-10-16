from __future__ import annotations
from vstools import core, set_output,vs,split,join
from typing import Optional, Any

__all__ = [
    'Stackable',
    'StackableManager',
    'oX',
    'oY',
    'oN',
    'fltr_to_expr',
    'fltr_to_expr_causal'
]

class Stackable:
    mode: int 
    
    #mode 0 expr
    #mode 1 plane
    #mode 2 operation/constant
    #mode 3 plane absolute address
    #mode 4 plane prop
    contents: Any
    #: list[Stackable] | tuple[str,int,int] | str | list[Stackable] | tuple[str,str]

#    def __key__(self):
#        return (mode,contents)
#
#    def replace(self, src_hash: int, dest: Stackable):
#        if hash(self) == src_hash:
#            return dest
#        if mode in [ 1, 2,3,4 ]:
#            return self
#        elif mode == 0:
#            return Stackable(0, [ a.replace(src_hash,dest) for a in self.contents ])
#        else:
#            assert False

    def __init__(self,mode,contents):
        if mode == 0:
            assert isinstance(contents,list)
            for a in contents:
                assert isinstance(a,Stackable)
        elif mode == 1:
            assert isinstance(contents,tuple)
            assert isinstance(contents[0],str)
            assert isinstance(contents[1],int)
            assert isinstance(contents[2],int)
        elif mode == 2:
            assert isinstance(contents,str)
        elif mode == 3:
            assert isinstance(contents,list)
            assert isinstance(contents[0],Stackable)# x
            assert isinstance(contents[1],Stackable)# y
            assert isinstance(contents[2],Stackable)# plane
            assert contents[2].mode == 1
        elif mode == 4:
            assert isinstance(contents,tuple)
            assert isinstance(contents[0],str)
            assert isinstance(contents[1],str)
        else:
            assert False
        self.mode = mode
        self.contents = contents

    @staticmethod
    def const(v: int | float) -> Stackable:
        return Stackable(2, str(v))


    @staticmethod
    def new_plane() -> Stackable:
        import random
        import string
        name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(30))
        return Stackable(1, (name,0,0))

    @staticmethod
    def plane(name: str) -> Stackable:
        return Stackable(1, (name,0,0))

    @staticmethod
    def operation(name: str) -> Stackable:
        return Stackable(2, name)

    @staticmethod
    def const_or_stack(v: int | float | Stackable):
        if isinstance(v,Stackable):
            return v
        else:
            return Stackable.const(v)
    def eval_vs(self, mapping_in: list[tuple[Stackable, vs.VideoNode]]):
        mapping = {}
        mapping_v = []

        i = 0
        assert len(mapping_in) >= 1
        for a in mapping_in:
            assert isinstance(a[0], Stackable)
            assert isinstance(a[1], vs.VideoNode)
            assert a[0].mode == 1
            mapping[a[0].contents[0]] = f"src{i}"
            mapping_v += [ a[1] ]
            i += 1
        stra = self.eval(mapping)
        return core.akarin.Expr(mapping_v, stra)

    def eval(self,mapping: dict):
        ret = ""
        if self.mode == 0:
            for a in self.contents:
                if isinstance(a, Stackable):
                    ret += a.eval(mapping) + " "
                elif isinstance(a,int) or isinstance(a,float):
                    ret += str(a) + " "
                else:
                    print(type(a))
                    assert False
        elif self.mode == 1:
            nam = self.contents[0]
            if nam in mapping:
                nam = mapping[nam]
            ret = f"{nam}[{self.contents[1]},{self.contents[2]}]"
        elif self.mode == 2:
            ret = self.contents
        elif self.mode == 3:
            assert self.contents[2].mode == 1
            c2c = self.contents[2].contents
            nam = c2c[0]
            if nam in mapping:
                nam = mapping[nam]
            ret = Stackable(0,[
                (self.contents[0] + c2c[1]),
                (self.contents[1] + c2c[2]),
                Stackable.operation(f"{nam}[]"),
            ]).eval(mapping)
        elif self.mode == 4:
            nam = self.contents[0]
            if nam in mapping:
                nam = mapping[nam]
            ret = f"{nam}.{self.contents[1]}"
        else:
            assert False
            
        return ret

    def __getitem__(self, item) -> Stackable:
        #relative
        if isinstance(item,str):
            assert self.mode == 1
            return Stackable(4, (self.contents[0],item))
        assert isinstance(item, tuple)
        if len(item) == 2:
            assert isinstance(item, tuple)
            assert [ isinstance(item[0],int), isinstance(item[1],int) ] == [True,True]
            assert len(item) == 2
            if self.mode == 0:
                return Stackable(0, [a[item] for a in self.contents])
            elif self.mode == 1:
                return Stackable(1, (self.contents[0], self.contents[1] + item[0], self.contents[2] + item[1]))
            elif self.mode == 2:
                return self
            elif self.mode == 3:
                return Stackable(3, [a[item] for a in self.contents])
            elif self.mode == 4:
                return self
            else:
                assert  False
        #absolute
        elif len(item) == 3:
            assert item[2] == "a"
            item = item[:-1]
            assert len(item) == 2
            assert [
                isinstance(item[0], Stackable) or isinstance(item[0],int),
                isinstance(item[1], Stackable) or isinstance(item[1],int)
            ] == [True,True]
            assert self.mode == 1
            return Stackable(3, [ 
                Stackable.const_or_stack(item[0]),
                Stackable.const_or_stack(item[1]),
                self,
             ])
        assert False
    
    def iftrue(self, then, elsee):
        return Stackable(0,[self, Stackable.const_or_stack(then), Stackable.const_or_stack(elsee), Stackable(2, "?")])
    
    def switch(self, a):
        assert isinstance(a,list)
        ret = Stackable.const(0.0)
        for v in a:
            assert isinstance(v,tuple) and len(v) == 2
            ret = ret + (self == v[0]).iftrue(v[1],0)
        return ret

    @staticmethod
    def choose_smallest(sm, a: list[tuple[Stackable, Stackable]]):
        if len(a) == 1:
            return a[0][1]

        opl = []

        opl += [ a[0][0], Stackable.operation("cmp_s!") ]
        opl += [ a[0][1], Stackable.operation("val_s!") ]

        for b in a[1:]:
            opl += [ b[0], b[1] ]
        for _ in a[1:]:
            opl += [ Stackable.operation("val! cmp!") ]
            opl += [ Stackable.operation("cmp@ cmp_s@ < val@ val_s@ ? val_s!") ]
            opl += [ Stackable.operation("cmp@ cmp_s@ < cmp@ cmp_s@ ? cmp_s!") ]
        opl += [ Stackable.operation("val_s@") ]
        return Stackable(0,opl)

    def sin(self):
        return Stackable(0,[self, Stackable(2, "sin")])
    def cos(self):
        return Stackable(0,[self, Stackable(2, "cos")])
    def abs(self):
        return Stackable(0,[self, Stackable(2, "abs")])
    def clip(self, mina, maxa):
        return Stackable(0,[self, Stackable.const_or_stack(mina),Stackable.const_or_stack(maxa),Stackable(2, "clip")])
    def max(self, other):
        return Stackable(0,[self, Stackable.const_or_stack(other),Stackable(2, "max")])
    def min(self, other):
        return Stackable(0,[self, Stackable.const_or_stack(other),Stackable(2, "min")])
    def clamp(self, mina, maxa):
        return Stackable(0,[self, Stackable.const_or_stack(mina), Stackable.const_or_stack(maxa),Stackable(2, "clamp")])
    def __or__(self, other):
        return Stackable(0,[self, Stackable.const_or_stack(other),Stackable(2, "or")])
    def __and__(self, other):
        return Stackable(0,[self, Stackable.const_or_stack(other),Stackable(2, "and")])
    def __neg__(self):
        return self * (-1)
    def __add__(self, other):
        return Stackable(0,[self, Stackable.const_or_stack(other), Stackable(2, "+")])
    def __sub__(self, other):
        return Stackable(0,[self, Stackable.const_or_stack(other), Stackable(2, "-")])
    def __mul__(self, other):
        return Stackable(0,[self, Stackable.const_or_stack(other), Stackable(2, "*")])
    def __truediv__(self, other):
        return Stackable(0,[self, Stackable.const_or_stack(other), Stackable(2, "/")])
    def __mod__(self, other):
        return Stackable(0,[self, Stackable.const_or_stack(other), Stackable(2, "%")])
    def __lt__(self, other):
        return Stackable(0,[self, Stackable.const_or_stack(other), Stackable(2, "<")])
    def __le__(self, other):
        return Stackable(0,[self, Stackable.const_or_stack(other), Stackable(2, "<=")])
    def __gt__(self, other):
        return Stackable(0,[self, Stackable.const_or_stack(other), Stackable(2, ">")])
    def __ge__(self, other):
        return Stackable(0,[self, Stackable.const_or_stack(other), Stackable(2, ">=")])
    def __eq__(self, other):
        return Stackable(0,[self, Stackable.const_or_stack(other), Stackable(2, "=")])
    def __ne__(self, other):
        return Stackable(0,[self, Stackable.const_or_stack(other), Stackable(2, "!=")])

oX = Stackable.operation("X")
oY = Stackable.operation("Y")
oN = Stackable.operation("N")

class StackableManager:
    all_mapping: list[tuple[Stackable,vs.VideoNode]]
    force_format: Optional[int]
    def __init__(self, formaty: Optional[int] = None):
        self.all_mapping = []
        self.force_format = formaty

    def add_clips(self, *args) -> list[Stackable]:
        return [ self.add_clip(a) for a in args]
    def add_clip(self, c: vs.VideoNode, name: Optional[str] = None) -> Stackable:
        if self.force_format is not None:
            assert c.format.id == self.force_format
        for a in self.all_mapping:
            assert a[1].width == c.width
            assert a[1].height == c.height
        assert isinstance(c,vs.VideoNode)
        import random
        import string
        if name is None:
            name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=30))

        if len(self.all_mapping) != 0:
            assert self.all_mapping[0][1].format.id == c.format.id 
        sta = Stackable.plane(name)
        self.all_mapping += [ (sta, c) ]
        return sta

    def eval_x(self, s: Stackable | list[Stackable]) -> Stackable | tuple[Stackable,...]:
        if isinstance(s,list):
            return tuple([self.eval(a)[0] for a in s])
        else:
            return self.eval(s)[0]

    def eval_v(self, s: Stackable) -> vs.VideoNode:
        assert isinstance(self.eval(s)[1],vs.VideoNode)
        return self.eval(s)[1]

    def eval(self, s: Stackable) -> tuple[Stackable, vs.VideoNode]:
        v = s.eval_vs(self.all_mapping)
        new_x = self.add_clip(v)

        self.all_mapping += [(new_x,v)]
        return new_x,v


def fltr_to_expr(x: Stackable,v: list[float]) -> Stackable:
    assert isinstance(x,Stackable)
    assert len(v) % 2 == 1
    n = (len(v) - 1) // 2

    sta = Stackable.const(0)
    for i,n in enumerate(range(-n,n+1)):
        sta += x[n,0] * v[i]
    
    return sta


def fltr_to_expr_causal(x: Stackable,v: list[float]) -> Stackable:
    assert isinstance(x,Stackable)

    sta = Stackable.const(0)
    for i in range(0,len(v)):
        sta += x[-i,0] * v[len(v)-1-i]
    return sta