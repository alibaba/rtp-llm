import os, sys, time, asyncio, threading, logging
from maga_transformer.pipeline.pipeline import Pipeline
from maga_transformer.model_factory import AsyncModel, ModelFactory, ModelConfig
from maga_transformer.utils.util import WEIGHT_TYPE

import pandas as pd
try: 
    import model_prof as mp
except:
    print("model_prof has been installed yet.")
    if int(os.getenv("PROFILE", "0")) > 0:
        raise "Installing git@gitlab.alibaba-inc.com:AliNPU/model_prof.git if Profiling with Asys/Nsys"
    class mp:
        def prof_start():
            pass
        def prof_stop():
            pass
from model_factory import *
import torch

origin_prompt = ["\n\nHuman: THE idea that some ethnic groups may, on average, be more intelligent than others is one of those hypotheses that dare not speak its name. But Gregory Cochran, a noted scientific iconoclast, is prepared to say it anyway. He is that rare bird, a scientist who works independently of any institution. He helped popularise the idea that some diseases not previously thought to have a bacterial cause were actually infections, which ruffled many scientific feathers when it was first suggested. And more controversially still, he has suggested that homosexuality is caused by an infection. Even he, however, might tremble at the thought of what he is about to do. Together with Jason Hardy and Henry Harpending, of the University of Utah, he is publishing, in a forthcoming edition of the Journal of Biosocial Science, a paper which not only suggests that one group of humanity is more intelligent than the others, but explains the process that has brought this about. The group in question are Ashkenazi Jews. The process is natural selection. Ashkenazim generally do well in IQ tests, scoring 12-15 points above the mean value of 100, and have contributed disproportionately to the intellectual and cultural life of the West, as the careers of Freud, Einstein and Mahler, pictured above, affirm. They also suffer more often than most people from a number of nasty genetic diseases, such as Tay-Sachs and breast cancer. These facts, however, have previously been thought unrelated. The former has been put down to social effects, such as a strong tradition of valuing education. The latter was seen as a consequence of genetic isolation. Even now, Ashkenazim tend to marry among themselves. In the past they did so almost exclusively. Dr Cochran, however, suspects that the intelligence and the diseases are intimately linked. His argument is that the unusual history of the Ashkenazim has subjected them to unique evolutionary pressures that have resulted in this paradoxical state of affairs. Ashkenazi history begins with the Jewish rebellion against Roman rule in the first century AD. When this was crushed, Jewish refugees fled in all directions. The descendants of those who fled to Europe became known as Ashkenazim. In the Middle Ages, European Jews were subjected to legal discrimination, one effect of which was to drive them into money-related professions such as banking and tax farming which were often disdained by, or forbidden to, Christians. This, along with the low level of intermarriage with their gentile neighbours (which modern genetic analysis confirms was the case), is Dr Cochran's starting point. He argues that the professions occupied by European Jews were all ones that put a premium on intelligence. Of course, it is hard to prove that this intelligence premium existed in the Middle Ages, but it is certainly true that it exists in the modern versions of those occupations. Several studies have shown that intelligence, as measured by IQ tests, is highly correlated with income in jobs such as banking. What can, however, be shown from the historical records is that European Jews at the top of their professions in the Middle Ages raised more children to adulthood than those at the bottom. Of course, that was true of successful gentiles as well. But in the Middle Ages, success in Christian society tended to be violently aristocratic (warfare and land), rather than peacefully meritocratic (banking and trade). Put these two things together—a correlation of intelligence and success, and a correlation of success and fecundity—and you have circumstances that favour the spread of genes that enhance intelligence. The questions are, do such genes exist, and what are they if they do? Dr Cochran thinks they do exist, and that they are exactly the genes that cause the inherited diseases which afflict Ashkenazi society. That small, reproductively isolated groups of people are susceptible to genetic disease is well known. Constant mating with even distant relatives reduces genetic diversity, and some disease genes will thus, randomly, become more common. But the very randomness of this process means there should be no discernible pattern about which disease genes increase in frequency. In the case of Ashkenazim, Dr Cochran argues, this is not the case. Most of the dozen or so disease genes that are common in them belong to one of two types: they are involved either in the storage in nerve cells of special fats called sphingolipids, which form part of the insulating outer sheaths that allow nerve cells to transmit electrical signals, or in DNA repair. The former genes cause neurological diseases, such as Tay- Sachs, Gaucher's and Niemann-Pick. The latter cause cancer.  \n\nHuman: Sure, let me continue this topic. What would you like to discuss? "]

def test_pipeline_time(pipeline, **kwargs):
    context_time = 0
    acc_time = 0
    token_count = 0
    exception = None

    async def generator():
        nonlocal context_time
        nonlocal acc_time
        nonlocal exception
        nonlocal token_count
        mp.prof_start()
        try:
            begin_time = time.time()
            torch.cuda.nvtx.range_push(f"Token-{token_count}")
            async for x in pipeline.pipeline_async(**kwargs):
                torch.cuda.nvtx.range_pop
                end_time = time.time()
                cost_time = end_time - begin_time
                if token_count == 0:
                    context_time = cost_time
                else:
                    acc_time += cost_time
                begin_time = end_time
                token_count += 1
                torch.cuda.nvtx.range_push(f"Token-{token_count}")
        except Exception as e:
            exception = e
            torch.cuda.nvtx.range_pop
        mp.prof_stop()
        

    def start_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(generator())

    backgroud_thread = threading.Thread(target=start_loop)
    backgroud_thread.start()
    backgroud_thread.join()

    if exception is not None:
        raise exception
    return context_time, acc_time/token_count


def main():
    log_level = os.getenv("LOG_LEVEL", "WARNING").upper()
    logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    print(" ----- Hello ----")
    tokenizer_path = os.environ["TOKENIZER_PATH"]
    ckpt_path = os.environ["CHECKPOINT_PATH"]
    model_type = os.environ["MODEL_TYPE"]
    tokenizer_merge_file_path = os.environ.get("TOKENIZER_MERGE_FILE_PATH", None)
    int8_mode = int(os.environ.get("INT8_MODE", "0"))
    async_mode = os.environ.get("AYSNC_MODE", None) == '1'

    model_args = {
        "model_type": model_type,
        "ckpt_path": ckpt_path,
        "tokenizer_path": tokenizer_path,
        "weight_type":  WEIGHT_TYPE.INT8 if int8_mode == 1 else WEIGHT_TYPE.FP16,
        "async_mode": async_mode, 
        "max_seq_len": 0
    }

    model = ModelFactory.from_model_type(ModelConfig(**model_args))
    if isinstance(model, AsyncModel):
        tokenizer = model.model.tokenizer
    else:
        tokenizer = model.tokenizer
    pipeline = Pipeline(model, tokenizer)

    test_batchsize = eval(os.environ.get("TEST_BATCHSIZE", "[1]"))
    test_input_len = eval(os.environ.get("TEST_INPUT_LEN", "[128]"))
    test_output_len = eval(os.environ.get("TEST_OUTPUT_LEN", "[10]"))

    result_list = list()
    for bs, seqlen, new_tokens in zip(test_batchsize, test_input_len, test_output_len):
        print(f"--- batchsize={bs}, seqlen={seqlen}, output={new_tokens}, INT8_MODE={int8_mode} ---")
        test_pipeline_time(
            pipeline=pipeline,
            prompts=origin_prompt*bs, images = [[]] * bs, top_k=1, top_p=0, temperature=1, context_len=seqlen,
            generate_config={'max_new_tokens': 10})        
        context_time, generate_time = test_pipeline_time(
            pipeline=pipeline,
            prompts=origin_prompt*bs, images = [[]] * bs, top_k=1, top_p=0, temperature=1, context_len=seqlen,
            generate_config={'max_new_tokens': new_tokens})
        print(f"context_time={context_time*1000:.3f} ms, generate_time={generate_time*1000:.3f} ms")
        result_list.append({"batchsize": bs, "seqlen": seqlen, "output": new_tokens, "INT8_MODE": int8_mode, "context_time": context_time*1000, "generate_time": generate_time*1000})

    if isinstance(pipeline.model, AsyncModel):
        pipeline.model.decoder_engine_.stop()

    del pipeline
    torch.cuda.empty_cache()

    result_df = pd.DataFrame(data=result_list)
    result_df["Generate_throughput"] = result_df["batchsize"]*1000/result_df["generate_time"]
    print(result_df)
    csv_path = os.environ.get('TEST_OUTPUT_DIR', '.')
    precision_mode= "fpAIntB" if int8_mode == 1 else "fp16"
    output_path = os.path.join(csv_path, f"{model_type}_{precision_mode}_perf_data.csv")
    result_df.to_csv(output_path)
    print(f"Perf Data written to {os.path.abspath(output_path)}")


if __name__ == '__main__':
    main()
   

