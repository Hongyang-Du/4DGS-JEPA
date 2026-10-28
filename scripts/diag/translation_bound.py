import torch
for s in ["jumpingjacks", "hellwarrior", "mutant", "standup"]:
    ck = torch.load(f"results/dnerf/{s}/fair/runs/jepa_full/last.pt", weights_only=False)
    st = ck["stats"]; print(s, "vmax %.5f  2*vmax*17 = %.3f  2*vmax*37 = %.3f  vel_scale %.5f cv_k %s" % (st.vmax, 2*st.vmax*17, 2*st.vmax*37, st.vel_scale, ck["cfg"].get("cv_k")))
