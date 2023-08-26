from PIL import Image

from diffuser_scripts.pipelines.adetailer import PipelineKeeper


prompt = 'ouyangnana[SEP] baijingting'
images = []
images.append(Image.open("test_result/couple_035_more/唐嫣_彭于晏_0.jpg"))
images.append(Image.open("test_result/couple_035_more/唐嫣_彭于晏_1.jpg"))
images.append(Image.open("test_result/couple_035_more/唐嫣_彭于晏_2.jpg"))

pk = PipelineKeeper()
pk.init_pipeline(2)
results = pk.process(images,"A [SEP] B","blablabla")
for i, r in enumerate(results):
    r.save(str(i) + '.jpg')