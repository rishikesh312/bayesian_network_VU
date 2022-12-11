
import os
import random
import time
import traceback
import uuid
import pandas as pd

from BNReasoner import BNReasoner

class BnrExperiment:
    def __init__(self, network_files, output_dir) -> None:
        self.network_files = network_files
        self.output_dir = output_dir
        self.result = []

    def run(self):
        for file in self.network_files:
            self.br = BNReasoner(file)
            file_name = file.split("/")[-1].split(".BIFXML")[0]
            print(file_name)
            vars = self.br.bn.get_all_variables()
            try:
                for i in range(100):
                    q, e = self.qe_generate(vars)

                    print(q, e)

                    map = self.br.map
                    mpe = self.br.mpe

                    self.run_inference(file_name, len(vars), map, q, e, prune=True, heuristic="fill", save_result=True)
                    self.run_inference(file_name, len(vars), map, q, e, prune=False, heuristic="fill", save_result=True)
                    self.run_inference(file_name, len(vars), map, q, e, prune=True, heuristic="degree", save_result=True)
                    self.run_inference(file_name, len(vars), map, q, e, prune=False, heuristic="degree", save_result=True)

                    self.run_inference(file_name, len(vars), mpe, q, e, prune=True, heuristic="fill", save_result=True)
                    self.run_inference(file_name, len(vars), mpe, q, e, prune=False, heuristic="fill", save_result=True)
                    self.run_inference(file_name, len(vars), mpe, q, e, prune=True, heuristic="degree", save_result=True)
                    self.run_inference(file_name, len(vars), mpe, q, e, prune=False, heuristic="degree", save_result=True)
                    
                    print("Done one Q&e with {}".format(file_name))
            except:
                print(traceback.format_exc())
                output_file = os.path.join(self.output_dir, file_name+".csv")
                self.write_result(output_file)

            output_file = os.path.join(self.output_dir, file_name+".csv")
            self.write_result(output_file)
            self.result = []
            # break

    def run_inference(self, network_file, netwrok_size, func, q, e, prune, heuristic, save_result=True):
        """
        Run one inference
        """
        self.br.resume_network()
        start = time.perf_counter()
        if func == self.br.map:
            result = func(q, e, prune, heuristic)
            func_name = "map"
        elif func == self.br.mpe:
            result = func(e, prune, heuristic)
            func_name = "mpe"
        else:
            raise NotImplementedError
        end = time.perf_counter()

        if save_result:
            self.result.append(
                self.wrap_result(
                    ["id", "network", "size", "query", "evidence", "method", "prune", "heuristic", "result", "time"], 
                    [uuid.uuid1(), network_file, netwrok_size, q, e, func_name, str(prune), heuristic, result[0].to_dict(), end-start])
            )
        else:
            return result, end-start

    def qe_generate(self, vars: list, seed=None):
        """
        Generate random queries and evidences
        """
        # random select n (also random) sub-vars(len less than 1/3 len of vars) from vars
        select_vars = random.sample(vars, random.randint(2, 5+len(vars)//10))
        # random select n (also random) sub-vars from selected vars as q
        select_q = random.sample(select_vars, random.randint(1, len(select_vars)-1))
        select_e = [one for one in select_vars if one not in select_q]
        e_dict = {}
        # random assign true/false value to e
        for var in select_e:
            e_dict[var] = random.choice([True, False])
        return select_q, e_dict

    def wrap_result(self, keys, vals):
        """
        Wrap indefence results as a dict
        """
        result_dict = {}
        for key, val in zip(keys, vals):
            result_dict[key] = val
        return result_dict

    def write_result(self, outfile):
        keys = list(self.result[0].keys())
        values = [list(one.values()) for one in self.result]

        df = pd.DataFrame(data=values, columns=keys)
        df.to_csv(outfile, ",")


if __name__ == "__main__":

    part = 1

    test_files = os.path.join(os.getcwd(), "testing")
    output_dir = os.path.join(os.getcwd(), "experiments")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = os.listdir(test_files)
    files.sort()

    files = [os.path.join(test_files, file) for file in files if "0.BIFXML" in file and "100" not in file]

    if part == 2:
        files = files[-5:]
    else:
        files = files[5:6]

    # print(files)

    he = BnrExperiment(files, output_dir)
    he.run()


