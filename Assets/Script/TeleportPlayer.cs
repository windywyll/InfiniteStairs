using UnityEngine;
using System.Collections;

public class TeleportPlayer : MonoBehaviour {

    [SerializeField]
    private GameObject teleportPoint;

    private float yMove;

    void OnTriggerEnter(Collider _col)
    {
        if(_col.transform.root.gameObject.layer == LayerMask.NameToLayer("Player"))
        {
            _col.transform.root.Translate(new Vector3(0, yMove, 0));
        }
    }

	// Use this for initialization
	void Start () {
        yMove = teleportPoint.transform.position.y - transform.position.y;
	}
	
	// Update is called once per frame
	void Update () {
	
	}
}
