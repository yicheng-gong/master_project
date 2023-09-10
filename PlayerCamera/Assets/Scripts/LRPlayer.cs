using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LRPlayer : MonoBehaviour
{
    public RenderTexture LRTexture;
    private Material material;
    // Start is called before the first frame update
    void Start()
    {
        material = new Material(Shader.Find("Unlit/Texture"));
        GetComponent<Renderer>().material = material;
    }
    void Update()
    {
        material.mainTexture = LRTexture;
    }
}
