from ..utils import common_functions as c_f
from .metric_loss_only import MetricLossOnly


class TrainWithClassifier(MetricLossOnly):
    def calculate_loss(self, curr_batch):
        data, labels = curr_batch
        embeddings,interm = self.compute_embeddings(data)
        intermediate, intermediate_cls =interm
        logits = self.maybe_get_logits(embeddings)
        indices_tuple = self.maybe_mine_embeddings(embeddings, labels)
        #intermediate_loss = self.maybe_get_metric_loss(intermediate, labels)
        indices_tuple_2= self.maybe_mine_embeddings(intermediate, labels)
        intermediate_loss = self.maybe_get_metric_loss(intermediate, labels, indices_tuple_2)

        intermediate_cls_loss =self.maybe_get_classifier_loss(intermediate_cls, labels)
        #indices_tuple_2= self.maybe_mine_embeddings()
        self.losses["metric_loss"] = self.maybe_get_metric_loss(
            embeddings, labels, indices_tuple
        )+0.2*intermediate_loss
        self.losses["classifier_loss"] = self.maybe_get_classifier_loss(logits, labels) + 0.2 *intermediate_cls_loss

    def maybe_get_classifier_loss(self, logits, labels):
        if logits is not None:
            return self.loss_funcs["classifier_loss"](
                logits, c_f.to_device(labels, logits)
            )
        return 0

    def maybe_get_logits(self, embeddings):
        if (
            self.models.get("classifier", None)
            and self.loss_weights.get("classifier_loss", 0) > 0
        ):
            return self.models["classifier"](embeddings)
        return None

    def modify_schema(self):
        self.schema["models"].keys += ["classifier"]
        self.schema["loss_funcs"].keys += ["classifier_loss"]
