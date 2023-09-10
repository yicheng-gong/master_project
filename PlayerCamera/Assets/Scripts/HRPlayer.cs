using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using System.Linq;
using System;
using UnityEngine.InputSystem;
using Unity.Barracuda.ONNX;
using UnityEditor;
using Unity.VisualScripting;
using UnityEngine.Profiling;
using Newtonsoft.Json.Linq;
using UnityEngine.Experimental.Rendering;

public class HRPlayer : MonoBehaviour
{
    public int scale_number;
    public RenderTexture LRTexture;
    public RenderTexture HRTexture;
    public int resolutionWeight;
    public int resolutionHeight;
    private Model m_RuntimeModel;
    private IWorker m_RuntimeWorker;
    private Material material;
    private RenderTexture outTexture;
    private Tensor HRTensor;
    
    // Start is called before the first frame update
    void Start()
    {
        string model_path = "model_" + scale_number.ToString() + "_200_" + resolutionWeight.ToString() + "_" + resolutionHeight.ToString();
        m_RuntimeModel = ModelLoader.Load((NNModel)Resources.Load(model_path));
        m_RuntimeWorker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, m_RuntimeModel);

        material = new Material(Shader.Find("Unlit/Texture"));
        GetComponent<Renderer>().material = material;

        outTexture = new RenderTexture(scale_number * resolutionWeight, scale_number * resolutionHeight, 0, GraphicsFormat.R8G8B8A8_SRGB);
        HRTensor = new(1, scale_number * LRTexture.height, scale_number * LRTexture.width, 3);
    }

    // Update is called once per frame
    void Update()
    {
        Tensor LRTensor = new(LRTexture, 3);
        // Execute the model
        // Profiler.BeginSample("Model");
        m_RuntimeWorker.Execute(LRTensor);
        HRTensor = m_RuntimeWorker.PeekOutput();
        LRTensor.Dispose();
        outTexture = HRTensor.ToRenderTexture(bias:0.25f);
        // Profiler.EndSample();
        Graphics.Blit(outTexture, HRTexture);
        material.mainTexture = HRTexture;
        HRTensor.Dispose();
        outTexture.Release();
    }
}
