using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CanvasPosTest : MonoBehaviour
{
    public Canvas testCanvas;

    private void Update()
    {
        if(Input.GetMouseButtonDown(0))
        {
            Debug.Log(Input.mousePosition);
            Debug.Log(testCanvas.ScreenToCanvasPosition(Input.mousePosition));
            Debug.Log(testCanvas.renderingDisplaySize);
        }
    }
}
