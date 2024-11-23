"""Script para la creacion del diagrama de la arquitectura."""

from diagrams import Diagram, Cluster, Edge
from diagrams.aws.analytics import Glue, Quicksight
from diagrams.aws.compute import LambdaFunction
from diagrams.aws.general import InternetAlt2, User, Users
from diagrams.aws.ml import SagemakerNotebook, SagemakerTrainingJob, SagemakerModel
from diagrams.aws.network import APIGateway
from diagrams.aws.storage import S3
from diagrams.custom import Custom

with Diagram("Arquitectura", show=False):
    consumers = Users("Consumidores")
    analysts = Users("Analistas")
    mle = User("MLE")

    source = InternetAlt2("Australia Meteorology site")
    github = Custom("GitHub", "docs/github_logo.png")

    with Cluster("Desarrollo"):
        sm_note = SagemakerNotebook("SM Studio Classic")
        sm_tjob = SagemakerTrainingJob("SM Training y HPO Jobs")
        sm_model = SagemakerModel("XGBoost endpoint")

        lambda_f = LambdaFunction("Invoke Function")
        api_gw = APIGateway("API Gateway")

    with Cluster("Storage"):
        data_s3 = S3("Data")
        glue = Glue("Glue DataBrew")
        results_s3 = S3("Predicciones")

    with Cluster("Analytics"):
        qs = Quicksight("Dashboard")

    mle >> sm_note >> sm_tjob >> sm_model
    sm_note >> Edge() << github
    (consumers, mle) >> Edge() << api_gw >> Edge() << lambda_f >> Edge() << sm_model
    analysts >> Edge() << qs

    source >> glue >> data_s3 >> sm_tjob
    sm_model >> results_s3 >> qs
