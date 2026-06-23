import multiprocessing
import time
from fastai.learner import Learner
from pymoo.operators.repair.vtype import TypeRepair
from torchvision import transforms
from pymoo.core.problem import ElementwiseProblem, Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.sampling.lhs import LHS
from pymoo.core.result import Result
from PIL import Image, ImageDraw
import torch
from mestradolib.mask import *
from mestradolib.model import ModelWrapper
from mestradolib.utils import *
from mestradolib.inpaint import *
from pymoo.optimize import minimize
import numpy as np
import sys


class IntegerLHS(LHS):
    def _do(self, problem, n_samples, **kwargs):
        samples = super()._do(problem, n_samples, **kwargs)
        return np.round(samples).astype(int)


class EvaluationObject:
    def __init__(
        self,
        n_var,
        n_obj,
        xl=None,
        xu=None,
        mutation=None,
        crossover=None,
        problem_best_image=None,
        problem_heatmap=None,
        problem_probability_heatmap=None,
        problem_name=None,
        dynamic_variables=False,
    ):
        self.n_var = n_var
        self.n_obj = n_obj
        self.xu = xu
        self.xl = xl
        self.mutation = mutation or PM(
            prob=1.0, eta=3, vtype=int, repair=RoundingRepair()
        )
        self.crossover = crossover or SBX(
            prob=1.0, eta=3, vtype=int, repair=RoundingRepair()
        )
        self.problem_best_image = problem_best_image
        self.problem_heatmap = problem_heatmap
        self.problem_probability_heatmap = problem_probability_heatmap
        self.problem_name = problem_name
        self.dynamic_variables = dynamic_variables

    def add_img(self, img: Image.Image):
        pass

    def evaluate(
        self,
        x,
        out,
        img: Image.Image,
        prob_img_original: float,
        pred_idx_original: int,
        model: ModelWrapper,
        *args,
        **kwargs,
    ):
        pass

    def get_mutation(self):
        return self.mutation

    def get_crossover(self):
        return self.crossover


def minimize_rectangle_problem(
    img: Image.Image,
    evaluation_object: EvaluationObject,
    model_wrapper: ModelWrapper,
    prob_img_original,
    pred_idx_original,
    n_proccess=8,
) -> Result:

    problem = VectorizedRectangleProblem(
        img,
        prob_img_original,
        pred_idx_original,
        model_wrapper,
        evaluation_object,
    )

    algorithm = NSGA2(
        pop_size=200,
        sampling=IntegerRandomSampling(),
        crossover=evaluation_object.get_crossover(),
        mutation=evaluation_object.get_mutation(),
    )

    res = minimize(
        problem, algorithm, ("n_gen", 10), seed=42, verbose=False, save_history=True
    )

    return res


class MenorProbalidadeMenorArea(EvaluationObject):
    def evaluate(
        self,
        x,
        out,
        img: Image.Image,
        prob_img_original: float,
        pred_idx_original: int,
        model: ModelWrapper,
        *args,
        **kwargs,
    ):
        if x[0] >= x[2] or x[1] >= x[3]:
            out["F"] = [sys.maxsize, sys.maxsize]
            return

        copied_image = img.copy()
        copied_image.paste("black", (x[0], x[1], x[2], x[3]))
        _, _, outputs = model.predict(copied_image)
        prob_img_com_retangulo = outputs[pred_idx_original].item()

        out["F"] = [
            prob_img_com_retangulo - prob_img_original,
            (x[2] - x[0]) * (x[3] - x[1]),
        ]


class RectangleProblem(ElementwiseProblem):
    def __init__(
        self,
        img: Image.Image,
        prob_img_original: float,
        pred_idx_original: int,
        model: ModelWrapper,
        evaluation_object: EvaluationObject,
        *args,
        **kwargs,
    ):
        super().__init__(
            n_var=evaluation_object.n_var,
            n_obj=evaluation_object.n_obj,
            n_ieq_constr=0,
            xl=0,
            xu=img.width if evaluation_object.xu is None else evaluation_object.xu,
            vtype=int,
            *args,
            **kwargs,
        )
        self.img = img
        self.model = model
        self.prob_img_original = prob_img_original
        self.pred_idx_original = pred_idx_original
        self.evaluation_object = evaluation_object

    def _evaluate(self, x, out, *args, **kwargs):
        self.evaluation_object.evaluate(
            x,
            out,
            self.img,
            self.prob_img_original,
            self.pred_idx_original,
            self.model,
            *args,
            **kwargs,
        )


class VectorizedRectangleProblem(Problem):
    def __init__(
        self,
        img: Image.Image,
        prob_img_original: float,
        pred_idx_original: int,
        model: ModelWrapper,
        evaluation_object: EvaluationObject,
        *args,
        **kwargs,
    ):
        xu = (
            max(img.width, img.height)
            if evaluation_object.xu is None
            else evaluation_object.xu
        )
        if callable(xu):
            xu = xu(img)

        xl = 0 if evaluation_object.xl is None else evaluation_object.xl
        if callable(xl):
            xl = xl(img)

        super().__init__(
            n_var=evaluation_object.n_var,
            n_obj=evaluation_object.n_obj,
            n_ieq_constr=0,
            xl=xl,
            xu=xu,
            vtype=int,
            *args,
            **kwargs,
        )
        self.img = img
        self.model = model
        self.prob_img_original = prob_img_original
        self.pred_idx_original = pred_idx_original
        self.evaluation_object = evaluation_object

    def _evaluate(self, x, out, *args, **kwargs):
        self.evaluation_object.evaluate(
            x,
            out,
            self.img,
            self.prob_img_original,
            self.pred_idx_original,
            self.model,
            *args,
            **kwargs,
        )


class ApenasMenorProbabilidade(EvaluationObject):
    def evaluate(
        self,
        x,
        out,
        img: Image.Image,
        prob_img_original: float,
        model: ModelWrapper,
        *args,
        **kwargs,
    ):
        if x[0] >= x[2] or x[1] >= x[3]:
            out["F"] = [sys.maxsize]
            return

        copied_image = img.copy()
        copied_image.paste("black", (x[0], x[1], x[2], x[3]))
        _, pred_idx_c, outputs = model.predict(copied_image)
        prob_img_com_retangulo = outputs[pred_idx_c].item()

        out["F"] = [prob_img_com_retangulo - prob_img_original]


class ProbabilidadeMaisArea(EvaluationObject):
    def evaluate(
        self,
        x,
        out,
        img: Image.Image,
        prob_img_original: float,
        model: ModelWrapper,
        *args,
        **kwargs,
    ):
        if x[0] >= x[2] or x[1] >= x[3]:
            out["F"] = [sys.maxsize]
            return

        copied_image = img.copy()
        copied_image.paste("black", (x[0], x[1], x[2], x[3]))
        _, pred_idx_c, outputs = model.predict(copied_image)
        prob_img_com_retangulo = outputs[pred_idx_c].item()

        area = (x[2] - x[0]) * (x[3] - x[1])
        relative_area = area / (img.width * img.height)

        out["F"] = [(prob_img_com_retangulo - prob_img_original) + relative_area]


class PoligonoEPerimetro(EvaluationObject):
    def evaluate(
        self,
        x,
        out,
        img: Image.Image,
        prob_img_original: float,
        pred_idx_original: int,
        model: ModelWrapper,
        *args,
        **kwargs,
    ):
        polygon_points = [(x[i], x[i + 1]) for i in range(0, len(x), 2)]

        copied_image = img.copy()
        draw = ImageDraw.Draw(copied_image)
        draw.polygon(polygon_points, fill="black")

        _, _, outputs = model.predict(copied_image)
        prob_img_com_retangulo = outputs[pred_idx_original].item()

        perimeter = 0
        for i in range(len(polygon_points)):
            j = (i + 1) % len(polygon_points)
            perimeter += (
                (polygon_points[i][0] - polygon_points[j][0]) ** 2
                + (polygon_points[i][1] - polygon_points[j][1]) ** 2
            ) ** 0.5

        out["F"] = [prob_img_com_retangulo - prob_img_original, perimeter]


class PoligonoSemIntercessaoEPerimetro(EvaluationObject):
    def evaluate(
        self,
        x,
        out,
        img: Image.Image,
        prob_img_original: float,
        model: ModelWrapper,
        *args,
        **kwargs,
    ):
        if polygon_has_crossing_edges(x):
            out["F"] = [sys.maxsize, sys.maxsize]
            return

        polygon_points = [(x[i], x[i + 1]) for i in range(0, len(x), 2)]

        copied_image = img.copy()
        draw = ImageDraw.Draw(copied_image)
        draw.polygon(polygon_points, fill="black")

        _, pred_idx_c, outputs = model.predict(copied_image)
        prob_img_com_retangulo = outputs[pred_idx_c].item()

        perimeter = 0
        for i in range(len(polygon_points)):
            j = (i + 1) % len(polygon_points)
            perimeter += (
                (polygon_points[i][0] - polygon_points[j][0]) ** 2
                + (polygon_points[i][1] - polygon_points[j][1]) ** 2
            ) ** 0.5

        out["F"] = [prob_img_com_retangulo - prob_img_original, perimeter]


class MenorProbalidadeMenorAreaInpaint(EvaluationObject):
    def evaluate(
        self,
        x,
        out,
        img: Image.Image,
        prob_img_original: float,
        pred_idx_original: int,
        model: ModelWrapper,
        *args,
        **kwargs,
    ):
        if x[0] >= x[2] or x[1] >= x[3]:
            out["F"] = [sys.maxsize, sys.maxsize]
            return

        copied_image = inpaint_image_with_rectangle(img, (x[0], x[1], x[2], x[3]))
        _, _, outputs = model.predict(copied_image)
        prob_img_com_retangulo = outputs[pred_idx_original].item()

        out["F"] = [
            prob_img_com_retangulo - prob_img_original,
            (x[2] - x[0]) * (x[3] - x[1]),
        ]


class MenorProbalidadeMenorAreaInpaintVectorized(EvaluationObject):
    def evaluate(
        self,
        x,
        out,
        img: Image.Image,
        prob_img_original: float,
        pred_idx_original: int,
        model: ModelWrapper,
        *args,
        **kwargs,
    ):
        images = [inpaint_image_with_rectangle(img, xi) for xi in x]
        prob_imgs_com_retangulo = model.get_vetorized_probabilities(
            images, pred_idx_original
        )

        areas = (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
        result = np.column_stack([prob_imgs_com_retangulo - prob_img_original, areas])
        result[(x[:, 0] >= x[:, 2]) | (x[:, 1] >= x[:, 3])] = [sys.maxsize, sys.maxsize]
        out["F"] = result


class MenorProbalidadeMenorAreaInpaintInverseVectorized(EvaluationObject):
    def evaluate(
        self,
        x,
        out,
        img: Image.Image,
        prob_img_original: float,
        pred_idx_original: int,
        model: ModelWrapper,
        *args,
        **kwargs,
    ):
        images = [inpaint_inverse_image_with_rectangle(img, xi) for xi in x]
        prob_imgs_com_retangulo = model.get_vetorized_probabilities(
            images, pred_idx_original
        )

        areas = (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
        result = np.column_stack([prob_img_original - prob_imgs_com_retangulo, areas])
        result[(x[:, 0] >= x[:, 2]) | (x[:, 1] >= x[:, 3])] = [sys.maxsize, sys.maxsize]
        out["F"] = result


class MenorProbalidadeMenorAreaVectorized(EvaluationObject):
    def evaluate(
        self,
        x,
        out,
        img: Image.Image,
        prob_img_original: float,
        pred_idx_original: int,
        model: ModelWrapper,
        *args,
        **kwargs,
    ):
        images = [mask_rectangle(img, xi) for xi in x]
        prob_imgs_com_retangulo = model.get_vetorized_probabilities(
            images, pred_idx_original
        )

        areas = (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
        result = np.column_stack([prob_imgs_com_retangulo - prob_img_original, areas])
        result[(x[:, 0] >= x[:, 2]) | (x[:, 1] >= x[:, 3])] = [sys.maxsize, sys.maxsize]
        out["F"] = result


class MenorProbalidadeMenorAreaInverseVectorized(EvaluationObject):
    def evaluate(
        self,
        x,
        out,
        img: Image.Image,
        prob_img_original: float,
        pred_idx_original: int,
        model: ModelWrapper,
        *args,
        **kwargs,
    ):
        images = [mask_inverse_rectangle(img, xi) for xi in x]
        prob_imgs_com_retangulo = model.get_vetorized_probabilities(
            images, pred_idx_original
        )

        areas = (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
        result = np.column_stack([prob_img_original - prob_imgs_com_retangulo, areas])
        result[(x[:, 0] >= x[:, 2]) | (x[:, 1] >= x[:, 3])] = [sys.maxsize, sys.maxsize]
        out["F"] = result


class IPHAFlavioMarceloInverseVectorized(EvaluationObject):
    def __init__(self, n_var, n_obj, xl=None, xu=None, mutation=None, crossover=None):
        super().__init__(n_var, 1, 0, 1, mutation, crossover)

    def evaluate(
        self,
        x,
        out,
        img: Image.Image,
        prob_img_original: float,
        pred_idx_original: int,
        model: ModelWrapper,
        *args,
        **kwargs,
    ):
        images = [mask(img, xi) for xi in x]
        prob_imgs_com_mascara = model.get_vetorized_probabilities(
            images, pred_idx_original
        )

        result = prob_img_original - prob_imgs_com_mascara
        out["F"] = result


class IPHAFlavioMarceloVectorized(EvaluationObject):
    def __init__(
        self,
        mutation=None,
        crossover=None,
        problem_best_image=None,
        problem_heatmap=None,
        problem_probability_heatmap=None,
        problem_name=None,
    ):
        super().__init__(
            0,
            1,
            1,
            2**16,
            mutation,
            crossover,
            problem_best_image,
            problem_heatmap,
            problem_probability_heatmap,
            problem_name,
            dynamic_variables=True,
        )

    def evaluate(
        self,
        x,
        out,
        img: Image.Image,
        prob_img_original: float,
        pred_idx_original: int,
        model: ModelWrapper,
        *args,
        **kwargs,
    ):
        masks = [mask_from_integers(xi, img.size) for xi in x]
        images = [mask(img, m) for m in masks]
        prob_imgs_com_mascara = model.get_vetorized_probabilities(
            images, pred_idx_original
        )
        out["F"] = prob_imgs_com_mascara - prob_img_original


class PoligonoEPerimetroVectorized(EvaluationObject):
    def evaluate(
        self,
        x,
        out,
        img: Image.Image,
        prob_img_original: float,
        pred_idx_original: int,
        model: ModelWrapper,
        *args,
        **kwargs,
    ):
        images = [mask_polygon(img, xi) for xi in x]
        prob_imgs_com_poligono = model.get_vetorized_probabilities(
            images, pred_idx_original
        )

        perimetros = calcula_perimetros(x)
        result = np.column_stack(
            [prob_imgs_com_poligono - prob_img_original, perimetros]
        )
        out["F"] = result


class DynamicPoligonoEPerimetroVectorized(EvaluationObject):
    def __init__(
        self,
        n_var,
        n_obj,
        xl=None,
        xu=None,
        mutation=None,
        crossover=None,
        intersection=True,
    ):
        super().__init__(n_var, n_obj, xl, xu, mutation, crossover)
        self.intersection = intersection

    def evaluate(
        self,
        x,
        out,
        img: Image.Image,
        prob_img_original: float,
        pred_idx_original: int,
        model: ModelWrapper,
        *args,
        **kwargs,
    ):
        # tfms = model.dls.valid.after_item

        model_torch = model.model
        model_torch.eval()
        model_torch.to(device=model.dls.device)

        # 1. Gerar todas as imagens modificadas

        images = [dynamic_mask_polygon(img, xi) for xi in x]

        dl = model.dls.test_dl(images)
        images_tensor = dl.one_batch()[0]
        images_tensor = images_tensor.to(device=model.dls.device).float()

        prob_imgs_com_retangulo = []
        with torch.no_grad():
            outputs = model_torch(images_tensor)
            prob_imgs_com_retangulo = torch.softmax(outputs, dim=1)

        prob_imgs_com_retangulo = (
            prob_imgs_com_retangulo[:, pred_idx_original].cpu().numpy()
        )

        perimetros = dynamic_calcula_perimetros(x)
        result = np.column_stack(
            [prob_imgs_com_retangulo - prob_img_original, perimetros]
        )

        if not self.intersection:
            mask = np.apply_along_axis(
                dynamic_polygon_has_crossing_edges, axis=1, arr=x
            ).astype(bool)
            result[mask] = [sys.maxsize, sys.maxsize]

        out["F"] = result


class PoligonoEPerimetroInpaintVectorized(EvaluationObject):
    def evaluate(
        self,
        x,
        out,
        img: Image.Image,
        prob_img_original: float,
        pred_idx_original: int,
        model: ModelWrapper,
        *args,
        **kwargs,
    ):
        images = [inpaint_image_with_points(img, xi) for xi in x]
        prob_imgs_com_poligono = model.get_vetorized_probabilities(
            images, pred_idx_original
        )

        perimetros = calcula_perimetros(x)
        result = np.column_stack(
            [prob_imgs_com_poligono - prob_img_original, perimetros]
        )
        out["F"] = result


class DynamicPoligonoEPerimetroInpaintVectorized(EvaluationObject):
    def __init__(
        self,
        n_var,
        n_obj,
        xl=None,
        xu=None,
        mutation=None,
        crossover=None,
        intersection=True,
    ):
        super().__init__(n_var, n_obj, xl, xu, mutation, crossover)
        self.intersection = intersection

    def evaluate(
        self,
        x,
        out,
        img: Image.Image,
        prob_img_original: float,
        pred_idx_original: int,
        model: ModelWrapper,
        *args,
        **kwargs,
    ):
        # tfms = model.dls.valid.after_item

        model_torch = model.model
        model_torch.eval()
        model_torch.to(device=model.dls.device)

        # 1. Gerar todas as imagens modificadas

        images = [dynamic_inpaint_image_with_points(img, xi) for xi in x]

        dl = model.dls.test_dl(images)
        images_tensor = dl.one_batch()[0]
        images_tensor = images_tensor.to(device=model.dls.device).float()

        prob_imgs_com_retangulo = []
        with torch.no_grad():
            outputs = model_torch(images_tensor)
            prob_imgs_com_retangulo = torch.softmax(outputs, dim=1)

        prob_imgs_com_retangulo = (
            prob_imgs_com_retangulo[:, pred_idx_original].cpu().numpy()
        )

        perimetros = dynamic_calcula_perimetros(x)
        result = np.column_stack(
            [prob_imgs_com_retangulo - prob_img_original, perimetros]
        )

        if not self.intersection:
            mask = np.apply_along_axis(
                dynamic_polygon_has_crossing_edges, axis=1, arr=x
            ).astype(bool)
            result[mask] = [sys.maxsize, sys.maxsize]

        out["F"] = result


class PoligonoEPerimetroPositiveVectorized(EvaluationObject):
    def evaluate(
        self,
        x,
        out,
        img: Image.Image,
        prob_img_original: float,
        pred_idx_original: int,
        model: ModelWrapper,
        *args,
        **kwargs,
    ):
        # tfms = model.dls.valid.after_item

        model_torch = model.model
        model_torch.eval()
        model_torch.to(device=model.dls.device)

        # 1. Gerar todas as imagens modificadas

        images = [positive_mask_polygon(img, xi) for xi in x]

        dl = model.dls.test_dl(images)
        images_tensor = dl.one_batch()[0]
        images_tensor = images_tensor.to(device=model.dls.device).float()

        prob_new_imgs = []
        with torch.no_grad():
            outputs = model_torch(images_tensor)
            prob_new_imgs = torch.softmax(outputs, dim=1)

        prob_new_imgs = prob_new_imgs[:, pred_idx_original].cpu().numpy()

        perimetros = calcula_perimetros(x)
        result = np.column_stack([prob_new_imgs * (-1), perimetros])
        out["F"] = result


class PoligonoEPerimetroComInpaint(EvaluationObject):
    def evaluate(
        self, x, out, img, prob_img_original, pred_idx_original, model, *args, **kwargs
    ):
        polygon_points = [(x[i], x[i + 1]) for i in range(0, len(x), 2)]

        inpainted_image = inpaint_image_with_polygon(img, polygon_points)

        _, _, outputs = model.predict(inpainted_image)
        prob_img_com_retangulo = outputs[pred_idx_original].item()

        perimeter = 0
        for i in range(len(polygon_points)):
            j = (i + 1) % len(polygon_points)
            perimeter += (
                (polygon_points[i][0] - polygon_points[j][0]) ** 2
                + (polygon_points[i][1] - polygon_points[j][1]) ** 2
            ) ** 0.5

        out["F"] = [prob_img_com_retangulo - prob_img_original, perimeter]


class PixelMask(EvaluationObject):
    def __init__(self, n_obj):
        super().__init__(n_var=0, n_obj=n_obj)

    def add_img(self, img):
        self.n_var = img.width * img.height
        self.xu = 1

    def evaluate(self, x, out, img, prob_img_original, model, *args, **kwargs):
        pass


class PixelMaskBlackMenorProbabilidadeMenorArea(PixelMask):
    def evaluate(self, x, out, img, prob_img_original, model, *args, **kwargs):
        mask = np.array(x, dtype=bool).reshape((img.height, img.width))
        copied_image = img.copy()
        arr = np.array(copied_image)
        arr[mask.T] = [0, 0, 0]
        copied_image = Image.fromarray(arr)

        _, pred_idx_c, outputs = model.predict(copied_image)
        prob_img_com_mask = outputs[pred_idx_c].item()

        out["F"] = [prob_img_com_mask - prob_img_original, np.sum(x)]


class PixelMaskInpaintMenorProbabilidadeMenorArea(PixelMask):
    def evaluate(self, x, out, img, prob_img_original, model, *args, **kwargs):
        mask = np.array(x, dtype=bool).reshape((img.height, img.width))
        copied_image = inpaint_image_with_mask(img, mask)

        _, pred_idx_c, outputs = model.predict(copied_image)
        prob_img_com_mask = outputs[pred_idx_c].item()

        out["F"] = [prob_img_com_mask - prob_img_original, np.sum(x)]
